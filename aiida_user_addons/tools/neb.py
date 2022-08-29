"""
Some tools for parsing data from NEB calculations
"""
from pathlib import Path
from typing import TextIO, Tuple, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from parsevasp.poscar import Poscar


def _parse_force_block(lines):
    """
    Parse the block of total forces from the OUTCAR file

    :param lines: A list of lines containing lines including the TOTAL-FORCE block

    :returns: A tuple of position and forces
    """
    forces = []
    positions = []
    istart = len(lines)
    for idx, line in enumerate(lines):
        if "TOTAL-FORCE (eV/Angst)" in line:
            istart = idx
        elif idx > istart + 1:
            if not line.startswith(" -----"):  # Still in the block
                values = list(map(float, line.split()))
                positions.append(values[:3])
                forces.append(values[3:])
            else:
                # Reached the end of the block
                break
    return positions, forces


def parse_neb_outputs(
    fpath_or_obj: Union[str, Path, TextIO]
):  # pylint: disable=too-many-branches,too-many-statements
    """
    Scan for NEB output in the OUTCAR content

    :param fpath_or_obj: Input path or fileobj

    :returns dict: A dictionary of the parsed data
    """
    if isinstance(fpath_or_obj, (str, Path)):
        with open(fpath_or_obj) as fobj:
            lines = fobj.readlines()
    # A file-like object
    elif hasattr(fpath_or_obj, "readlines"):
        # Reset seek
        fpath_or_obj.seek(0)
        lines = fpath_or_obj.readlines()
    else:
        raise ValueError(f"'fpath_or_obj' variable is not supported: {fpath_or_obj}")

    vtst_data = {}
    output = {"neb_converged": False}
    output["frames"] = []

    for idx, line in enumerate(lines):
        if "NIONS" in line:
            nions = int(line.split()[-1])

        elif "VTST: version" in line:
            vtst_data["version"] = line.split(":")[1].strip()

        elif "NEB: Tangent" in line:
            tangents = []
            for isub in range(idx + 2, idx + 99999):
                subline = lines[isub]
                if subline.strip():
                    tangents.append([float(tmp) for tmp in subline.split()])
                else:
                    break
            vtst_data["tangents"] = tangents
        elif "NEB: forces" in line:
            forces = [float(tmp) for tmp in line.split()[-3:]]
            vtst_data["force_par_spring"] = forces[0]
            vtst_data["force_perp_real"] = forces[1]
            vtst_data["force_dneb"] = forces[2]
        elif "stress matrix after NEB project" in line:
            stress = []
            for isub in range(idx + 1, idx + 4):
                stress.append([float(tmp) for tmp in lines[isub].split()])
            vtst_data["stress_matrix"] = stress
        elif "FORCES: max atom" in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data["force_max_atom"] = forces[0]
            vtst_data["force_rms"] = forces[1]
        elif "FORCE total and by dimension" in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data["force_total"] = forces[0]
            vtst_data["force_by_dimension"] = forces[1]
        elif "Stress total and by dimension" in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data["stress_total"] = forces[0]
            vtst_data["stress_by_dimension"] = forces[1]
        elif "OPT: skip step - force has converged" in line:
            vtst_data["neb_converged"] = True
        elif "energy(sigma->0)" in line:
            tokens = line.split()
            vtst_data["energy_extrapolated"] = float(tokens[-1])
            vtst_data["energy_without_entropy"] = float(tokens[-4])
        elif "free  energy   TOTEN" in line:
            vtst_data["free_energy"] = float(line.split()[-2])
        elif "TOTAL-FORCE" in line:
            positions, forces = _parse_force_block(lines[idx : idx + nions + 10])
            vtst_data["outcar-forces"] = np.array(forces)
            vtst_data["outcar-positions"] = np.array(positions)
        elif "direct lattice vectors" in line:
            cell = []
            for subline in lines[idx + 1 : idx + 4]:
                cell.append([float(tmp) for tmp in subline.split()[:3]])
            output["outcar-cell"] = np.array(cell)

        elif "LOOP+" in line:
            # Signals end of iteration
            vtst_data["LOOP+"] = float(line.strip().split()[-1])
            output["frames"].append(vtst_data)
            vtst_data = {}

    output["last_neb_data"] = vtst_data
    return output


def get_neb_trajectory(
    poscar: Union[str, Path, TextIO], outcar: Union[str, Path, TextIO]
):
    """
    Obtain the trajectory of NEB optimisation

    Returns two list of atoms, one for the effective projected forces and the other for the tangential forces
    """
    if isinstance(poscar, (str, Path)):
        pparser = Poscar(file_path=poscar)
    else:
        pparser = Poscar(file_handler=poscar)
    outcar_data = parse_neb_outputs(outcar)
    poscar_data = pparser.get_dict()
    sites = poscar_data["sites"]
    cell = poscar_data["unitcell"]
    species = [site["specie"] for site in sites]
    positions = np.stack([site["position"] for site in sites], axis=0)

    # Construct the bast atom
    base_atoms = Atoms(symbols=species, scaled_positions=positions, cell=cell, pbc=True)

    frames = []
    tangents = []
    excluded = ["tangents", "stress_matrix"]
    for frame in outcar_data["frames"]:
        new_atoms = Atoms(
            symbols=species,
            positions=frame["outcar-positions"],
            cell=frame.get("outcar-cell", cell),
            pbc=True,
        )
        new_atoms.info.update(
            {
                key: value
                for key, value in frame.items()
                if not key.startswith("outcar") and key not in excluded
            }
        )
        calc = SinglePointCalculator(
            atoms=new_atoms,
            energy=frame["energy_extrapolated"],
            forces=frame["outcar-forces"],
            stress=frame["stress_matrix"],
        )
        new_atoms.calc = calc

        tangent_atoms = new_atoms.copy()
        calc = SinglePointCalculator(
            atoms=tangent_atoms,
            energy=frame["energy_extrapolated"],
            forces=frame["tangents"],
            stress=frame["stress_matrix"],
        )
        tangent_atoms.calc = calc
        frames.append(new_atoms)
        tangents.append(tangent_atoms)
    return frames, tangents


def fix_going_through_pbc_(atoms: Atoms, atoms_ref: Atoms) -> Tuple[Atoms, np.ndarray]:
    """
    Reorder the atoms to that of the reference.

    Only works for identical or nearly identical structures that are ordered differently.
    Returns a new `Atoms` object with order similar to that of `atoms_ref` as well as the sorting indices.
    """

    # Find distances
    acombined = atoms_ref.copy()
    acombined.extend(atoms)
    new_index = []
    # Get piece-wise MIC distances
    jidx = list(range(len(atoms), len(atoms) * 2))  # Through the structure to be fixed
    for i in range(len(atoms)):
        dists = acombined.get_distances(i, jidx, mic=True)
        # Find the index of the atom with the smallest distance
        min_idx = np.where(dists == dists.min())[0][0]
        new_index.append(min_idx)
        # Fix the displacement - ensure that it is reference to the reference atom
        # Distance from the match site in the input atoms to that of the reference atom
        disp = acombined.get_distance(i, min_idx + len(atoms), mic=True, vector=True)
        # Position of the atom using the displacement vector and the original structure in the reference atoms
        acombined.positions[min_idx] = acombined.positions[i] + disp
    assert len(set(new_index)) == len(atoms), "The detected mapping is not unique!"
    return acombined[new_index], new_index


def get_energy_from_misc(misc):
    """Get energy from a MISC output node"""
    tot = misc["total_energies"]
    return tot["energy_extrapolated"]


def sorted_values(input):
    """Sorted values based on the keys of a input dictionary"""
    data = list(input.items())
    data.sort(key=lambda x: x[0])
    return [tmp[1] for tmp in data]


class NEBData:
    """Class for handling NEB data"""

    def __init__(
        self,
        neb_node,
        initial_output,
        final_output,
        initial_forces=None,
        final_forces=None,
    ):
        """Instantiate the data object"""
        self._initial_energy = get_energy_from_misc(initial_output)
        self._final_energy = get_energy_from_misc(final_output)
        self.neb_node = neb_node
        self.nimages = len(self.neb_node.inputs.neb_images)

        # Get the NEB images
        self._input_images = [
            x.get_ase() for x in sorted_values(self.neb_node.inputs.neb_images)
        ]
        self._initial_structure = self.neb_node.inputs.initial_structure.get_ase()
        # Fill with zero forces
        if initial_forces is None:
            initial_forces = np.zeros(self._initial_structure.positions.shape)
        self._initial_structure.calc = SinglePointCalculator(
            self._initial_structure, energy=self._initial_energy, forces=initial_forces
        )

        self._final_structure = self.neb_node.inputs.final_structure.get_ase()
        # Fill with zero forces
        if final_forces is None:
            final_forces = np.zeros(self._final_structure.positions.shape)
        self._final_structure.calc = SinglePointCalculator(
            self._final_structure, energy=self._final_energy, forces=final_forces
        )

        self.n_iters = []
        self._calcjobs = []

        # Parse trajectory for the calcultion
        # The trajectory include real (unprojected) forces and energies
        all_trajs = []
        for node in self.neb_node.called:
            if node.is_finished:
                trajs = self._parse_trajectory_from_calc(node)
                self.n_iters.append(len(trajs))
                all_trajs.extend(trajs)
                self._calcjobs.append(node)
        # Extend with initial/final images
        for iteration in all_trajs:
            iteration.append(self._final_structure)
            iteration.insert(0, self._initial_structure)

        self.trajectory = all_trajs

    @classmethod
    def init_from_neb_workchain(cls, workchain):
        """Initialise from NEB workchain and automatically workout the terminal energy/forces"""
        from aiida import orm

        initial_structure = workchain.inputs.initial_structure
        final_structure = workchain.inputs.final_structure

        # Find for the initial image
        q = orm.QueryBuilder()
        q.append(orm.Node, filters={"id": initial_structure.id})
        q.append(orm.CalcJobNode, with_outgoing=orm.Node, project=["*"])
        q.append(
            orm.Dict,
            with_incoming=orm.CalcJobNode,
            project=["*"],
            edge_filters={"label": "misc"},
        )
        calc, initial_misc = q.one()
        if "output_forces" in calc.outputs:
            initial_forces = calc.outputs.output_forces.get_array("final")
        else:
            initial_forces = None

        q = orm.QueryBuilder()
        q.append(orm.Node, filters={"id": final_structure.id})
        q.append(orm.CalcJobNode, with_outgoing=orm.Node, project=["*"])
        q.append(
            orm.Dict,
            with_incoming=orm.CalcJobNode,
            project=["*"],
            edge_filters={"label": "misc"},
        )
        if q.count() > 0:
            final_calc, final_misc = q.one()
        else:
            q = orm.QueryBuilder()
            q.append(orm.Node, filters={"id": final_structure.id})
            q.append(orm.CalcFunctionNode, with_outgoing=orm.Node)
            q.append(
                orm.Structure,
                with_outgoing=orm.CalcFunctionNode,
                edge_filters={"label": {"like": "final%"}},
            )
            q.append(orm.CalcJobNode, with_outgoing=orm.Node, project=["*"])
            q.append(orm.Dict, with_incoming=orm.CalcJobNode, project=["*"])
            final_calc, final_misc = q.one()
        if "output_forces" in final_calc.outputs:
            final_forces = final_calc.outputs.output_forces.get_array("final")
        else:
            final_forces = None
        return cls(workchain, initial_misc, final_misc, initial_forces, final_forces)

    def get_nebtool_obj(self, idx):
        """Return an NEBTool object"""
        from ase.neb import NEBTools

        traj = self.trajectory[idx]
        return NEBTools(traj)

    def _parse_trajectory_from_calc(self, calc):
        """Parse information from the calculation"""
        nimages = self.nimages
        trajs = []  # Indexed by the image number
        retrieved = calc.outputs.retrieved
        for i in range(nimages):
            idx = i + 1
            with calc.base.repository.open(f"{idx:02d}/POSCAR") as poscar:
                with retrieved.base.repository.open(f"{idx:02d}/OUTCAR") as outcar:
                    traj, _ = get_neb_trajectory(poscar, outcar)
                    trajs.append(traj)
        niter = min(map(len, trajs))
        for i, traj in enumerate(trajs):
            if len(traj) != niter:
                print(
                    f"WARNING - IMAGE {i} contains {len(traj)} images than valid iterations: {niter}."
                )

        # Make a per iteration indexed array
        # output[i] contains the images of the ith iteration
        output = []
        for i in range(niter):
            output.append([traj[i] for traj in trajs])
        return output
