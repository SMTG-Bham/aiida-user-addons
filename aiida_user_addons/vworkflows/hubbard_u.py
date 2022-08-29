"""
Workflow for obtaining U from the linear response method

Following the original VASP tutorial, the process is as follows

1. Identify the site of the interest
2. Make a supercell (that is large enough)
3. Modify the POSCAR and POTCAR ordering such that the marked field is a different specie
4. Perform a single point calculation with  `LDAU=.FALSE.`, `LORBIT=11`
5. Exact the occupation numbers, and store the WAVECAR and the CHGCAR
6. Run a non-self-consistent calculation (`ICHARG=11`) with `LDAUTYPE=3` (activate the bias potential).
7. Run a self-consistent calculation with `LDAUTYPE=3` (activate the bias potential).
8. Collect the occupation numbers for the site of the interest.
9. Repeat 6/7/8 for different values of the bias potential.
10. Fit the change in occupation nubmers vs the bias potential and extract the gradients


Note that VASP does not seems to allow having the `self-consistent` response, as
`LDAUTYPE` tags is used for both the `alpha` term and the `U/J` terms.
In principles this should be possible, by altering the source code?
"""
import aiida.orm as orm
import numpy as np
from aiida.engine import WorkChain, append_, calcfunction
from aiida.plugins import WorkflowFactory
from ase.build import make_supercell, sort

from ..common.opthold import (
    DictOption,
    IntOption,
    ListOption,
    OptionContainer,
)
from ..common.repository import open_compressed


class LinearResponseUOptions(OptionContainer):
    """Container for settings of the linear response U workchain"""

    l = IntOption("Angular momentum channel", 2)
    magmom_mapping = DictOption("Mapping for the magnetic moment")
    magmom = ListOption("Magnetic moment of each site in the supercell")
    sites = ListOption(
        "A list of sites in the input cell to be perturbed", required=True
    )
    supercell = ListOption(
        "Matrix or list specifying the supercell to be used", required=True
    )


class LinearResponseU(WorkChain):
    """
    Run a series of calculations to extract self-consistent U value
    """

    _base_workchain = WorkflowFactory("vaspu.vasp")

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(cls._base_workchain, "vasp", exclude=("structure",))
        spec.input("structure", valid_type=orm.StructureData)
        spec.input(
            "response_settings",
            valid_type=orm.Dict,
            help="Settings of the workchain",
            serializer=LinearResponseUOptions.serialise,
            validator=LinearResponseUOptions.validate_dict,
        )
        spec.outline(
            cls.initialize,  # internal parameters
            cls.make_supercell,  # Make the supercell with correct species
            cls.do_scf,  # Run the SCF calculation
            cls.inspect_scf,  # Inspect the SCF calculation
            cls.do_responses,  # Launch the non-scf and scf calculations with potentials
            cls.result,  # Process and store the final results
        )
        spec.output(
            "fitting_results",
            valid_type=orm.Dict,
            help="Fitted U value and fitting data.",
        )
        spec.exit_code(
            301, "ERROR_INITIAL_SCF_FAILED", message="The initial SCF is failed."
        )

    def initialize(self):
        self.ctx.response_settings = self.inputs["response_settings"].get_dict()
        self.ctx.structure_in = self.inputs["structure"]
        self.ctx.potential_mapping = None  # Update potcar mapping
        assert "supercell" in self.ctx.response_settings
        assert "sites" in self.ctx.response_settings

    def make_supercell(self):
        """Generate the supercell"""
        supercell = orm.List(list=self.ctx.response_settings["supercell"])
        sites = orm.List(list=self.ctx.response_settings["sites"])

        self.ctx.supercell_structure = create_supercell_with_tags(
            self.ctx.structure_in, supercell, sites
        )

        kind_names_orig = [kind.name for kind in self.ctx.structure_in.kinds]
        kinds_super = self.ctx.supercell_structure.kinds

        new_kind = [kind for kind in kinds_super if kind.name not in kind_names_orig]
        if len(new_kind) != 1:
            raise RuntimeError(
                f"There should be one and only one new kinds, but these are found {new_kind}"
            )
        new_kind = new_kind[0]

        # Update the potcar mapping to reflect the new sites
        potcar_mapping = self.inputs.vasp.potential_mapping.get_dict()
        potcar_mapping[new_kind.name] = potcar_mapping[new_kind.symbol]
        self.ctx.potential_mapping = orm.Dict(dict=potcar_mapping)

    def do_scf(self):
        """Perform SCF calculation"""
        inputs = self.exposed_inputs(self._base_workchain, "vasp")

        inputs.structure = self.ctx.supercell_structure
        inputs.potential_mapping = self.ctx.potential_mapping

        # Ensure that we DO NOT clean the workdir
        if "clean_workdir" in inputs and inputs.clean_workdir.value is True:
            inputs.clean_workdir = orm.Bool(False)

        param = inputs.parameters.get_dict()

        # Default magmoment for fully FM state
        default_magmom = [1.0 for _ in inputs.structure.sites]

        # If mapping is specified - use it
        if "magmom_mapping" in self.ctx.response_settings:
            mapping = self.ctx.response_settings["magmom_mapping"]
            default_magmom = [
                mapping[site.kind_name] for site in inputs.structure.sites
            ]

        incar_update = {
            "ldau": False,
            "lcharg": True,
            "lwave": True,
            "lmaxmix": param["incar"].get("lmixmax", 4),
            "lorbit": 11,
            "ibrion": -1,
            "nsw": 0,
            "ispin": 2,
            "magmom": self.ctx.response_settings.get("magmom", default_magmom),
        }
        param["incar"].update(incar_update)

        # NOTE need to enforce parsing of the site_occupations array from the OUTCAR
        # once it is implemented

        # Update the input if any change is made
        if param != inputs.parameters.get_dict():
            inputs.parameters = orm.Dict(dict=param)

        inputs.metadata.label = self.inputs.metadata.label + " INITIAL SCF"
        running = self.submit(self._base_workchain, **inputs)
        self.to_context(initial_scf=running)
        self.report(f"Submitted initial ground state SCF workchain {running}")

    def inspect_scf(self):
        """Inspect the initial SCF calculation"""
        workchain = self.ctx.get("initial_scf")
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_INITIAL_RELAX_FAILED

    def do_responses(self):
        """Run the response calculations"""
        alphas = self.ctx.response_settings.get("alphas", [-0.2, -0.1, 0.1, 0.2])
        nkinds = len(self.ctx.supercell_structure.kinds)
        ldaul = [-1] * nkinds
        # Update for the last specie (to be perturbed)
        ldaul[-1] = self.ctx.response_settings["l"]
        ldauj = [0.0] * nkinds
        ldauu = [0.0] * nkinds

        incar_update = {
            "ldau": True,
            "ldautype": 3,
            "ldaul": ldaul,
            "ldauu": ldauu,
            "ldauj": ldauj,
            "icharg": 11,
        }

        inputs = self.exposed_inputs(self._base_workchain, "vasp")

        inputs.structure = self.ctx.supercell_structure
        inputs.potential_mapping = self.ctx.potential_mapping
        scf_param = self.ctx.initial_scf.inputs.parameters.get_dict()

        for alpha in alphas:
            incar_update["ldauj"][-1] = alpha
            incar_update["ldauu"][-1] = alpha
            scf_param["incar"].update(incar_update)
            scf_param["incar"]["icharg"] = 11
            inputs.parameters = orm.Dict(dict=scf_param)
            inputs.metadata.label = (
                self.inputs.metadata.label + f" NONSCF ALPHA={alpha}"
            )
            # Link the restart folder
            inputs.restart_folder = self.ctx.initial_scf.outputs.remote_folder

            running = self.submit(self._base_workchain, **inputs)
            running.set_extra("alpha", alpha)
            self.report(f"Submitted {running} for NSCF ALPHA={alpha}")
            self.to_context(nscf_workchains=append_(running))

            # For the SCF response
            scf_param["incar"].pop(
                "icharg"
            )  # Use default ICHARG - this should reuse the wave function
            inputs.parameters = orm.Dict(dict=scf_param)
            inputs.metadata.label = self.inputs.metadata.label + f" SCF ALPHA={alpha}"
            running = self.submit(self._base_workchain, **inputs)
            running.set_extra("alpha", alpha)
            self.report(f"Submitted {running} for SCF ALPHA={alpha}")
            self.to_context(scf_workchains=append_(running))

        self.report(f"All response calculations have been submitted")

    def result(self):
        """
        Analyse the results
        """
        nonzero = [
            node
            for node in (self.ctx.scf_workchains + self.ctx.nscf_workchains)
            if not node.is_finished_ok
        ]
        if nonzero:
            self.report(
                f"Workchains: {nonzero} finished with error, hence will not be used for fitting."
            )

        lmap = {0: "s", 1: "p", 2: "d", 3: "f"}
        channel = lmap[self.ctx.response_settings["l"]]
        nsites = len(self.ctx.response_settings["sites"])

        base_occ = parse_charge_projection(self.ctx.initial_scf)[channel][-nsites:]

        # Process NSCF
        alphas_nscf = [0.0]
        occ_nscf = [base_occ]

        for work in self.ctx.nscf_workchains:
            if not work.is_finished_ok:
                continue
            projection = parse_charge_projection(work)
            # Assuming single site, for now
            occ_nscf.append(projection[channel][-nsites:])
            alphas_nscf.append(work.get_extra("alpha"))

        alphas_scf = [0.0]
        occ_scf = [base_occ]
        for work in self.ctx.scf_workchains:
            if not work.is_finished_ok:
                continue
            projection = parse_charge_projection(work)
            # Assuming single site, for now
            occ_scf.append(projection[channel][-nsites:])
            alphas_scf.append(work.get_extra("alpha"))

        occ_scf = np.array(occ_scf)
        alphas_nscf = np.array(alphas_nscf)
        alphas_scf = np.array(alphas_scf)
        occ_nscf = np.array(occ_nscf)

        scf_fit = np.polyfit(alphas_scf, occ_scf, deg=1)
        nscf_fit = np.polyfit(alphas_nscf, occ_nscf, deg=1)
        # The higher power is the first element
        fit_u = 1 / scf_fit[0, :] - 1 / nscf_fit[0, :]

        # We do not have it wrapped in a calcfunction for now....
        output = {
            "occ_scf": occ_scf.tolist(),
            "occ_nscf": occ_nscf.tolist(),
            "alphas_scf": alphas_scf.tolist(),
            "alphas_nscf": alphas_nscf.tolist(),
            "fit_u": fit_u.tolist(),
        }
        output = orm.Dict(dict=output)
        output.store()
        self.out("fitting_results", output)


@calcfunction
def create_supercell_with_tags(
    structure_in: orm.StructureData, supercell: orm.List, sites: orm.List
):
    """
    Create supercell of the original structure, with certain sites tagged as different species.

    For example, tagged O sites will be marked as O1.
    """
    atoms = structure_in.get_ase()

    supercell_spec = supercell.get_list()
    if isinstance(supercell_spec[0], list):
        supercell_spec = np.array(supercell_spec)
    else:
        supercell_spec = np.diag(supercell_spec)

    repeated = make_supercell(atoms, supercell_spec)

    tags = repeated.get_tags()
    for i in sites.get_list():
        tags[i] = 1
    repeated.set_tags(tags)

    # Tags for sorting - ensure that the tagged species in the end
    sort_tags = repeated.get_atomic_numbers()
    for i in sites.get_list():
        sort_tags[i] += 1000
    repeated = sort(repeated, sort_tags)

    structure_out = orm.StructureData(ase=repeated)
    structure_out.label = structure_in.label + " SUPERCELL"
    return structure_out


def read_charge_projection(lines):
    """
    Read out the charge projection from line of the OUTCAR file
    """
    charge_projection = None
    for (n, line) in enumerate(lines):
        if "total charge" == line.strip():
            charge_entries = []
            for i in range(n + 4, len(lines)):
                if lines[i].startswith("tot"):
                    ilast = i
            for subline in lines[n + 4 : ilast - 1]:
                if subline.startswith("----"):
                    break
                charge_entries.append([float(token) for token in subline.split()])

            index, s, p, d, tot = zip(*charge_entries)
            charge_projection = {
                "index": list(map(int, index)),
                "s": list(s),
                "p": list(p),
                "d": list(d),
                "tot": list(tot),
            }
    if charge_projection is None:
        raise RuntimeError("No charge projection is found")
    return charge_projection


def parse_charge_projection(calcjob):
    """
    Bespoke parsing of the outcar data from calcjob
    """
    folder = calcjob.outputs.retrieved
    with open_compressed(folder, "OUTCAR", "r") as fhandle:
        outcar = fhandle.readlines()
    return read_charge_projection(outcar)
