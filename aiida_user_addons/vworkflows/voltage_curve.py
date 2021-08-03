"""
Module for computing voltage curves

The calculation of a voltage curve is more involved than a simple "voltage" as it is affected
by the relative stability of the intermediate delithiated structures.

The basic logic is the following:

1. Identify the *delithiatable* part of the compsision
2. Genearte unique delithiated structures, up to a certain supercell size (given as inputs)
3. Perform geometry optimisation on all delithiated structures
4. Construct a convex hull and identify the stable phases
5. Compute voltages based on the calculated energies
6. Store the segments of the curve

"""
from re import L
from aiida.orm.querybuilder import QueryBuilder
import numpy as np

from pymatgen.core import Structure
import aiida.orm as orm
from aiida.common.links import LinkType
from aiida.engine import WorkChain, if_, append_, calcfunction
from aiida.plugins import WorkflowFactory
from pymatgen.core.composition import Composition
from sqlalchemy.orm.query import Query

from aiida_user_addons.process.transform import delithiate_by_wyckoff, delithiate_full, delithiate_unique_sites, rattle
from aiida_user_addons.process.battery import VoltageCurve, _obtain_li_ref_calc, compute_li_voltage_shortcut, check_li_ref_calc
from aiida_user_addons.common.inputset.vaspsets import get_ldau_keys
from aiida_user_addons.common.misc import get_energy_from_misc
from aiida_user_addons.tools.pymatgen import get_entry_from_calc

from .mixins import WithVaspInputSet
from .common import OVERRIDE_NAMESPACE

VaspRelaxWorkChain = WorkflowFactory('vaspu.relax')


class VoltageCurveWorkChain(WorkChain):
    """Workchain for compute voltage curves (or the necessary information to assemble one"""
    ALLOWED_OK_EXIT_CODES = [601]

    @classmethod
    def define(cls, spec):

        super().define(spec)

        spec.expose_inputs(VaspRelaxWorkChain, 'relax', exclude=('structure',))
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('rattle', valid_type=orm.Float, help='Amplitude of rattling', default=lambda: orm.Float(0.05))
        spec.input('final_li_level',
                   valid_type=orm.Float,
                   help='Final lithiation level with respect to the reduced formula of the non-Li part of the composition.')
        spec.input(
            'lithiated_calc_misc',
            valid_type=orm.Dict,
            required=False,
            help=
            'The misc output of a calculation for the fully lithiated phase. If passed, will not perform any relaxation for the initial lithiated structure.'
        )
        spec.input(
            'li_metal_calc_misc',
            valid_type=orm.Dict,
            required=True,
            help='The misc output of a reference calculation for Li metal',
        )
        spec.input('atol', required=False, valid_type=orm.Float, help='Symmetry torlerance for generating unique structures.')
        spec.input_namespace('unique_options', required=False, populate_defaults=False, help='Options for delithiate by unique sites')
        spec.input('unique_options.excluded_sites', required=False, default=lambda: orm.List(list=[]), help='Excluded site indices.')
        spec.input('unique_options.nsub', required=False, default=lambda: orm.Int(1), help='Number of Li to remove')
        spec.input('unique_options.atol', required=False, default=lambda: orm.Float(1e-5), help='Symmetry tolerance')
        spec.input('unique_options.limit', required=False, help='Maximum number of structures to attempt')
        spec.input(
            'ldau_mapping',
            required=True,  # I set it to be mandatory here as in most cases we will need a U
            help='Mapping for LDA+U, see `get_ldau_keys` function for details',
            valid_type=orm.Dict)
        spec.input('deli_magmom_mapping', required=True, help='Mapping for MAGMOM for the delithiated state', valid_type=orm.Dict)
        spec.outline(
            cls.setup,
            if_(cls.should_run_initial_relax)(
                cls.run_initial_relax,  # Run relaxation of the initial structure
                cls.verify_initial_relax,  # Check the relaxation of the initial structure
            ),
            cls.generate_delithiated_structures,  # Delithiate the structures
            cls.run_delithiated_relax,  # Deploy relaxation for the delithiated structures
            cls.result,  # Finialise the results
        )
        spec.output_namespace('relaxed_delithiated_structures',
                              dynamic=True,
                              valid_type=orm.StructureData,
                              required=True,
                              help='Relaxed delithiated structures')
        spec.output('relaxed_lithiated_structure', valid_type=orm.StructureData, required=False, help='Relaxed original structure')
        spec.output('voltage_curve_data', valid_type=orm.Dict, required=True, help='Summary data regarding the voltage curve.')

        spec.exit_code(301, 'ERROR_INITIAL_RELAX_FAILED', message='The initial relaxation is failed.')
        spec.exit_code(401,
                       'ERROR_SOME_SUB_WORK_FAILED',
                       message='Some of the launched relaxations have failed - results may still be usable.')
        spec.exit_code(501, 'ERROR_ALL_SUB_WORK_FAILED', message='All of the launched relaxations have failed.')

    def setup(self):
        """Setup the workspace"""
        self.ctx.delithiated_frames = {}  # delithiated structures
        self.ctx.delithiated_masks = {}  # masks for the delithiated structures
        self.ctx.workchains = []  # List of WorkChains for the underlying calculation
        self.ctx.current_structure = self.inputs.structure  # List of the current structure
        self.ctx.lithiated_calc = None

        # Store the reference calculation for Li metal
        self.ctx.li_metal_calc = get_creator(self.inputs.li_metal_calc_misc)

    def should_run_initial_relax(self):
        """Wether to run the initial relaxation"""
        if self.inputs.get('lithiated_calc_misc'):
            calc = get_creator(self.inputs.lithiated_calc_misc)
            self.ctx.lithiated_calc = calc
            return False
        return True

    def run_initial_relax(self):
        """VaspRelaxWorkChain the input structure"""
        inputs = self.exposed_inputs(VaspRelaxWorkChain, 'relax')
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'initial_relax'

        if not inputs.metadata.get('label'):
            inputs.metadata.label = inputs.structure.get_formula('count') + ' RELAX'

        running = self.submit(VaspRelaxWorkChain, **inputs)
        self.to_context(initial_relax=running)
        self.report(f'Submitted initial relaxation workchain {running}')

    def verify_initial_relax(self):
        """Verify if the initial relaxation has been a success"""
        workchain = self.ctx.get('initial_relax')

        # Stop if there was a problem
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_INITIAL_RELAX_FAILED

        try:
            self.ctx.current_structure = workchain.outputs.relax__structure  # WORKCHAIN specific!
        except AttributeError:
            self.ctx.current_structure = workchain.outputs.relax.structure  # WORKCHAIN specific!

        # Makes sure that the current structure is labelled - as it will be used in the next steps
        self.ctx.current_structure.label = workchain.inputs.structure.label + ' RELAXED'
        # Record the calculation for the lithiated structure
        self.ctx.lithiated_calc = get_creator(workchain.outputs.misc)

        self.report('Initial relaxation is a success, registerred the relaxed structure')
        self.out('relaxed_lithiated_structure', self.ctx.current_structure)

    def generate_delithiated_structures(self):
        """Delithiate the structure at a range of Li concentrations"""
        from aiida_user_addons.process.battery import create_delithiated_multiple_level
        structure = self.ctx.current_structure
        # Generate all of the delithiated frames
        if 'atol' in self.inputs:
            delithiated = create_delithiated_multiple_level(structure, self.inputs.final_li_level, self.inputs.rattle, self.inputs.atol)
        else:
            delithiated = create_delithiated_multiple_level(structure, self.inputs.final_li_level, self.inputs.rattle)

        self.ctx.delithiated_frames = delithiated

    def run_delithiated_relax(self):
        """Proceed the relaxation of the delithiated frames"""
        all_frames = self.ctx.delithiated_frames
        nst = len(all_frames)
        for istruc, (key, frame) in enumerate(all_frames.items()):

            # Now setup the calculations
            inputs = self.exposed_inputs(VaspRelaxWorkChain, 'relax')
            inputs.structure = frame  # Use the delithiated structure

            # Set the call link label to identify the relaxation for specific structures
            inputs.metadata.call_link_label = f'relax_{key}'

            # Label the frame by formula and iterations here we just use the label assigned
            # by the original calcfunction
            if not inputs.metadata.get('label'):
                inputs.metadata.label = frame.label

            # Setup the parameters - update the magnetic moments
            deli_magmom_mapping = self.inputs.deli_magmom_mapping.get_dict()
            param_dict = inputs.vasp.parameters.get_dict()

            # Apply the supplied mapping for delithiated structures
            magmom = []
            default = deli_magmom_mapping.get('default', 0.6)  # Default MAGMOM is 0.6
            for site in frame.sites:
                magmom.append(deli_magmom_mapping.get(site.kind_name, default))
            param_dict[OVERRIDE_NAMESPACE]['magmom'] = magmom

            # Setup LDA+U
            ldau_settings = self.inputs.ldau_mapping.get_dict()
            ldau_keys = get_ldau_keys(frame, **ldau_settings)
            param_dict[OVERRIDE_NAMESPACE].update(ldau_keys)

            # Set the parameters
            inputs.vasp.parameters = orm.Dict(dict=param_dict)

            # Submit the calculation - the order does not matter here
            running = self.submit(VaspRelaxWorkChain, **inputs)
            self.report(f'Submitted {running} for structure {istruc + 1} out of {nst}.')
            self.to_context(workchains=append_(running))

        self.report(f'All {nst} relaxations submitted')

    def result(self):
        """
        Analyse and summarise the results

        This step is called after all delithiation calculation are finished.
        Here, the voltage curve data are composed with stable structures recorded.
        """
        miscs = {}
        nfail = 0
        # Inspect all workchains
        for workchain in self.ctx.workchains:
            link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
            if not workchain.is_finished_ok and workchain.exit_status not in self.ALLOWED_OK_EXIT_CODES:
                self.report('Relaxation {} ({}) did not finished ok - not attaching the results'.format(workchain, link_name))
                nfail += 1
            else:
                miscs[link_name] = workchain.outputs.misc
                if workchain.exit_status in self.ALLOWED_OK_EXIT_CODES:
                    self.report('Relaxation {} ({}) finished with exit code: {}, but treated as if it was OK'.format(
                        workchain, link_name, workchain.exit_status))

        if not miscs:
            self.report('None of the work has finished ok - serious problem must occurred')
            return self.exit_codes.ERROR_ALL_SUB_WORK_FAILED

        # Include the lithiated results which is either from the initial relax or from the passed misc node
        miscs['lithiated'] = self.ctx.lithiated_calc.outputs.misc

        # Construct the record node
        record_node = compose_voltage_curve_data(self.inputs.li_metal_calc_misc, **miscs)
        self.out('voltage_curve_data', record_node)

        if nfail > 1:
            return self.exit_codes.ERROR_SOME_SUB_WORK_FAILED

        # Attached the relaxed structures
        frames = self.ctx.delithiated_frames
        for key, frame in frames.items():
            q = orm.QueryBuilder()
            q.append(orm.Node, filters={'id': frame.id})
            q.append(VaspRelaxWorkChain, filters={'attributes.exit_status': 0})
            q.append(orm.StructureData, edge_filters={'label': 'relax__structure'})
            try:
                relaxed = q.one()[0]
            except Exception:
                continue
            self.out(f'relaxed_delithiated_structures.{key}', relaxed)

        return


@calcfunction
def compose_voltage_curve_data(li_ref_misc, **miscs):
    """
    Compose voltage curve calculation results from a list of misc Dict nodes

    The supplied misc nodes should includes those of all the lithiated and the delithiated
    structures.
    """

    entries = []
    for _, misc in miscs.items():

        # Locate the workchain node it is the relaxation that returned this MISC
        q = orm.QueryBuilder()
        q.append(VaspRelaxWorkChain, project=['*'])
        q.append(orm.Node, filters={'id': misc.pk})
        workchain = q.one()[0]

        entry = get_entry_from_calc(workchain)
        entries.append(entry)

    # Locate the reference calculations
    q = orm.QueryBuilder()
    q.append(orm.CalculationNode, project=['*'])
    q.append(orm.Node, filters={'id': li_ref_misc.pk})
    ref_calc = q.one()[0]
    ref_entry = get_entry_from_calc(ref_calc)

    # Compose a dictionary of lists from a list of dictionaries
    vcurve = VoltageCurve(entries, ref_entry)
    stables = []
    for entry in vcurve.stable_entries:
        stables.append(entry.parameters['calc_uuid'])
    xvalues, yvalues = vcurve.get_plot_data(x_axis_deli=True)
    outdict = {
        'stable_calcs': stables,
        'voltage_curve_x': xvalues,
        'voltage_curve_y': yvalues,
    }
    voltage_pair_data = []
    for (pair, voltage) in vcurve.compute_voltages():
        voltage_pair_data.append({
            'comp1': pair[0].as_dict(),
            'comp2': pair[1].as_dict(),
            'voltage': voltage,
        })
    outdict['voltage_pair_data'] = voltage_pair_data
    return orm.Dict(dict=outdict)


def get_creator(node):
    """Return the creator CalculationNode that generated the passed node"""
    q = QueryBuilder()
    q.append(orm.CalculationNode, project=['*'])
    q.append(orm.Node, filters={'id': node.id})
    calc = q.one()[0]
    return calc


def get_returner(node, node_type):
    """Return the creator CalculationNode that generated the passed node"""
    q = QueryBuilder()
    q.append(node_type, project=['*'])
    q.append(orm.Node, filters={'id': node.id})
    calc = q.one()[0]
    return calc


def get_voltage_curve_obj(workchain):
    """Regenerate a VoltageCurve object from a completed workchain node"""
    if 'lithiated_calc_misc' in workchain.inputs:
        lithiated_calc = get_creator(workchain.inputs.lithiated_calc_misc)
    else:
        # Get the relaxation workchain that was used to create the lithiated structure
        lithiated_calc = get_returner(workchain.outputs.relaxed_lithiated_structure, VaspRelaxWorkChain)
    li_ref_calc = get_creator(workchain.inputs.li_metal_calc_misc)
    delithiated_calcs = []
    for node in workchain.outputs.relaxed_delithiated_structures.values():
        delithiated_calcs.append(get_returner(node, VaspRelaxWorkChain))

    li_ref_entry = get_entry_from_calc(li_ref_calc)
    lithiated_entry = [get_entry_from_calc(lithiated_calc)]
    delithiated_entries = [get_entry_from_calc(node) for node in delithiated_calcs]

    return VoltageCurve(lithiated_entry + delithiated_entries, li_ref_entry)
