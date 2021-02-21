"""
WorkChain to perform delithiation
"""
import numpy as np

import aiida.orm as orm
from aiida.common.links import LinkType
from aiida.engine import WorkChain, if_, append_, calcfunction
from aiida.plugins import WorkflowFactory

from aiida_user_addons.process.transform import delithiate_by_wyckoff, delithiate_full, delithiate_unique_sites, rattle
from aiida_user_addons.process.battery import compute_li_voltage_shortcut, check_li_ref_calc
from aiida_user_addons.common.inputset.vaspsets import get_ldau_keys

from .mixins import WithVaspInputSet
from .common import OVERRIDE_NAMESPACE

Relax = WorkflowFactory('vaspu.relax')


class SimpleDelithiateWorkChain(WorkChain, WithVaspInputSet):
    """
    Simple delithiation by removing sites,
    """

    _allowed_strategies = ['full', 'unique', 'wyckoff']

    @classmethod
    def define(cls, spec):

        super().define(spec)

        spec.expose_inputs(Relax, 'relax', exclude=('structure',))
        spec.input('li_ref_group_name',
                   valid_type=str,
                   non_db=True,
                   default='li-metal-refs',
                   help='Name of the group containing calculations of Li metal references.')
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('strategy', valid_type=orm.Str, help='Delithiation strategy to be used. Choose from: full, unique, wyckoff.')
        spec.input('rattle_amp',
                   valid_type=orm.Float,
                   help='Rattle the structure after delithiate',
                   required=False,
                   default=lambda: orm.Float(0.1))
        spec.input('run_initial_relax',
                   valid_type=orm.Bool,
                   default=lambda: orm.Bool(True),
                   required=False,
                   help='Whether to run the initial relaxation.')
        spec.input('wyckoff_sites', valid_type=orm.List, required=False, help='Wyckoff sites to remove for the wyckoff mode.')
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
            cls.delithiate,  # Delithiate the structures
            cls.run_delithiated_relax,  # Deploy relaxation for the delithiated structures
            cls.result,  # Finialise the results
        )
        spec.output_namespace('delithiated_structures',
                              dynamic=True,
                              valid_type=orm.StructureData,
                              required=True,
                              help='Delithiated structures generated for relaxation.')
        spec.output_namespace('relaxed_delithiated_structures',
                              dynamic=True,
                              valid_type=orm.StructureData,
                              required=True,
                              help='Relaxed delithiated structures')
        spec.output('relaxed_lithiated_structure', valid_type=orm.StructureData, required=False, help='Relaxed original structure')

        spec.output('output_parameters', valid_type=orm.Dict, required=True, help='Summary of the result of delithiation.')
        spec.output('output_voltages', valid_type=orm.Dict, required=False, help='Voltages computed from delithiation.')

        spec.exit_code(201, 'ERROR_UNKNOWN_STRATEGY', message='Strategy to delithiate is not known.')
        spec.exit_code(301, 'ERROR_INITIAL_RELAX_FAILED', message='The initial relaxation is failed.')
        spec.exit_code(401,
                       'ERROR_SOME_SUB_WORK_FAILED',
                       message='Some of the launched relaxations have failed - results may still be usable.')
        spec.exit_code(501, 'ERROR_ALL_SUB_WORK_FAILED', message='All of the launched relaxations have failed.')

    def setup(self):
        """Setup the workspace"""
        self.ctx.delithiated_frames = {}  # Delithiated structures
        self.ctx.delithiated_masks = {}  # Delithiated structures
        self.ctx.workchains = []  # List of WorkChains for the underlying calculation
        self.ctx.current_structure = self.inputs.structure  # List of the current structure
        self.ctx.initial_relax = None  # Initial relaxation workchain
        strategy = self.inputs.strategy.value
        if strategy not in self._allowed_strategies:
            self.report(f'Strategy: {strategy} is not valid, valid ones are: {self._allowed_strategies}')
            return self.exit_codes.ERROR_UNKNOWN_STRATEGY
        self.ctx.strategy = strategy

        # Check the existence of Li-metal reference calculations, do not proceed if there is no reference
        self.check_li_ref()

    def check_li_ref(self):
        """
        Check the existence if Li-ref calculations
        """
        inputs = self.exposed_inputs(Relax, 'relax')
        incar = inputs['parameters']['incar']

        gga = incar.get('gga')
        encut = incar.get('encut')

        if not check_li_ref_calc(incar.get('encut'), incar.get('gga'), self.inputs.li_ref_group_name):
            raise RuntimeError(f'Li reference calculations do not exist for GGA:{gga}, encut: {encut}')

    def should_run_initial_relax(self):
        """Wether to run the initial relaxation"""
        return self.inputs.run_initial_relax.value

    def run_initial_relax(self):
        """Relax the input structure"""
        inputs = self.exposed_inputs(Relax, 'relax')
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'initial_relax'
        running = self.submit(Relax, **inputs)
        self.to_context(initial_relax=running)
        self.report(f'Submitted initial relaxation workchain {running}')

    def verify_initial_relax(self):
        """Verify if the initial relaxation has been a success"""
        workchain = self.ctx.get('initial_relax')

        # Stop if there was a problem
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_INITIAL_RELAX_FAILED

        self.ctx.current_structure = workchain.outputs.relax__structure  # WORKCHAIN specific!
        self.report('Initial relaxation is a success, registerred the relaxed structure')
        self.out('relaxed_lithiated_structure', workchain.outputs.relax__structure)

    def delithiate(self):
        """Delithiate according to the strategy defined"""
        strategy = self.ctx.strategy
        structure = self.ctx.current_structure
        frames = []
        masks = []

        if strategy == 'full':
            result = delithiate_full(structure)
            frames.append(result['structure'])
            masks.append(result['mask'])  # Masks of the site that have been kept
        elif strategy == 'wyckoff':
            wyckoff_sites = self.inputs.wyckoff_sites.value
            for wsite in wyckoff_sites:
                result = delithiate_by_wyckoff(structure, orm.Str(wsite))
                frames.append(result['structure'])
                masks.append(result['mask'])
        elif strategy == 'unique':
            unique_options = self.inputs.unique_options
            result = delithiate_unique_sites(structure, **unique_options)
            nframes = int(len(result) / 2)
            frames = [None for i in range(nframes)]
            masks = [None for i in range(nframes)]
            for link_name, node in result.items():
                ltype, lid = link_name.split('_')
                lid = int(lid)
                if ltype == 'structure':
                    frames[lid] = node
                elif ltype == 'mapping':
                    masks[lid] = node
                else:
                    raise RuntimeError(f'Unknown link name detected {ltype}')
        self.report(f'Delithiated frames ({len(frames)}) have been created')

        ramp = self.inputs.rattle_amp.value

        # Rattle the frames if needed
        if ramp != 0.0:
            rattle_frames = [rattle(frame, self.inputs.rattle_amp) for frame in frames]
            frames = rattle_frames
            self.report(f'Rattled the frames with amplitude {ramp}')

        self.out_many({f'delithiated_structures.structure_{i:04d}': structure for i, structure in enumerate(frames)})
        self.report('Attached the created structures to outputs.delithiated_structures')

        self.ctx.delithiated_frames = frames
        self.ctx.delithiated_masks = masks

    def run_delithiated_relax(self):
        """Proceed the relaxation of the delithiated frames"""
        all_frames = self.ctx.delithiated_frames
        all_masks = self.ctx.delithiated_masks
        nst = len(all_frames)
        for istruc, (frame, mask) in enumerate(zip(all_frames, all_masks)):
            if isinstance(mask, orm.ArrayData):
                mapping = mask.get_array('site_mapping')
            elif isinstance(mask, orm.List):
                mapping = np.array(mask.get_list())
            # Now setup the calculations
            inputs = self.exposed_inputs(Relax, 'relax')
            inputs.structure = frame  # Use the delithiated structure

            # Set the call link label to identify the relaxation
            # Note this may not be the same as the actual link name
            inputs.metadata.call_link_label = f'relax_{istruc:02d}'
            # Label the frame by formula and iterations
            if not inputs.metadata.get('label'):
                inputs.metadata.label = frame.get_formula('count') + f' DELI {istruc:02d}'

            # Setup the parameters - update the magnetic moments
            deli_magmom_mapping = self.inputs.deli_magmom_mapping.get_dict()
            param_dict = inputs.vasp.parameters.get_dict()
            if not deli_magmom_mapping:
                self.report('WARNING: Empty mapping given for magmom - keeping the original')
                magmom = param_dict[OVERRIDE_NAMESPACE].get('magmom')
                # Keep the original MAGMOM used for relaxation
                if magmom:
                    magarray = np.array(magmom)
                    new_array = magarray[mapping]  # Use the mapping to get a new list of MAGMOM
                    new_magmom = new_array.tolist()
                    param_dict[OVERRIDE_NAMESPACE]['magmom'] = new_magmom
            else:
                # Apply the supplied mapping for relaxing delithiated structure
                magmom = []
                default = deli_magmom_mapping.get('default', 0.6)  # Default MAGMOM is 0.6
                for site in frame.sites:
                    magmom.append(deli_magmom_mapping.get(site.kind_name, default))
                param_dict[OVERRIDE_NAMESPACE]['magmom'] = magmom

            # Setup LDA+U
            ldau_settings = self.inputs.ldau_mapping.get_dict()
            ldau_keys = get_ldau_keys(frame, **ldau_settings)
            param_dict[OVERRIDE_NAMESPACE].update(ldau_keys)

            inputs.vasp.parameters = orm.Dict(dict=param_dict)

            # Submit the calculation - the order does not matter here
            running = self.submit(Relax, **inputs)
            self.report(f'Submitted {running} for structure {istruc + 1} out of {nst}.')
            self.to_context(workchains=append_(running))

        self.report(f'All {nst} relaxations submitted')

    def result(self):
        """Check the status of the relaxed structures"""
        miscs = {}
        nfail = 0
        for workchain in self.ctx.workchains:
            link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
            if not workchain.is_finished_ok:
                self.report('Relaxation {} ({}) did not finished ok - not attaching the results'.format(workchain, link_name))
                nfail += 1
            else:
                miscs[link_name] = workchain.outputs.misc

        if not miscs:
            self.report('None of the work has finished ok - serious problem must occurred')
            return self.exit_codes.ERROR_ALL_SUB_WORK_FAILED
        # Construct the record node
        record_node = compose_delithiation_data(**miscs)
        self.out('output_parameters', record_node)

        # Construct the voltage node
        voltage_node = self.get_voltages()
        if voltage_node:
            voltage_node.store()
            self.out('output_voltages', voltage_node)

        if nfail > 1:
            return self.exit_codes.ERROR_SOME_SUB_WORK_FAILED

        # Attached the relaxed structures
        frames = self.ctx.delithiated_frames
        for i, frame in enumerate(frames):
            q = orm.QueryBuilder()
            q.append(orm.Node, filters={'id': frame})
            q.append(Relax, filters={'attributes.exit_status': 0})
            q.append(orm.StructureData, edge_filters={'label': 'relax__structure'})
            try:
                relaxed = q.one()[0]
            except Exception:
                continue
            self.out(f'relaxed_delithiated_structures.structure_{i:04d}', relaxed)

        return

    def get_voltages(self):
        """Compute the voltages"""
        relax = self.ctx.initial_relax
        if not relax:
            self.report('Initial relaxation not performed - skipping voltage computation')
            return

        voltage_data = []
        for workchain in self.ctx.workchains:
            if not workchain.is_finished_ok:
                self.report(f'Skip failed workchain {workchain} for voltage calculation')
                continue
            try:
                voltage = compute_li_voltage_shortcut(relax, workchain, store_provenance=False)
            except RuntimeError:
                self.report('Cannot compute voltage - possibly the Li reference is not available')
                break
            link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
            ist = int(link_name.split('_')[-1])

            voltage_data.append({'relax_work': workchain.uuid, 'relax_id': ist, 'voltage': voltage})

        # No data is found
        if not voltage_data:
            return

        voltage_dict = transpose_dict(voltage_data)
        return orm.Dict(dict=voltage_dict)


@calcfunction
def compose_delithiation_data(**miscs):
    """
    Compose delithiation results from a list of misc Dict nodes
    the relaxation workchain should be called with a linke names in the format
    of <link>_<id>
    """

    allowed_fields = ['removed_specie', 'removed_wyckoff', 'removed_specie', 'removed_site', 'delithiate_id']
    records = []
    for _, misc in miscs.items():

        # Locate the workchain node it is the relaxation that returned this MISC
        q = orm.QueryBuilder()
        q.append(Relax, project=['*'])
        q.append(orm.Node, filters={'id': misc.pk})
        workchain = q.one()[0]

        link_name = workchain.get_incoming(link_type=LinkType.CALL_WORK).one().link_label
        ist = int(link_name.split('_')[-1])

        deli_struct = workchain.inputs.structure
        deli_pstruct = deli_struct.get_pymatgen()
        num_fu = deli_pstruct.composition.get_reduced_composition_and_factor()[1]

        # Find the link label of the delithiated structure
        deli_link = deli_struct.get_incoming(link_type=LinkType.CREATE).one().link_label

        misc = workchain.outputs.misc.get_dict()
        miscs[link_name] = misc  # Record the misc nodes for linking
        magnetisation = misc.get('magnetization', [])

        if magnetisation:
            total_mag = sum(magnetisation)
        else:
            total_mag = 0.0

        entry = {
            'relax_id': ist,  # ID of the relaxation launched by the workchain
            'relaxation': workchain.uuid,
            'output_structure': workchain.outputs.relax__structure.uuid,
            'input_structure': deli_struct.uuid,
            'energy': misc['total_energies']['energy_no_entropy'] / num_fu,
            'total_magnetisation': total_mag / num_fu,
            'magnetisation': magnetisation,
            'deli_link_label': deli_link,
        }
        # Add any available fields from the original structure
        attrs = deli_struct.attributes
        for attr in allowed_fields:
            if attr in attrs:
                entry[attr] = attrs[attr]

        records.append(entry)

    # Compose a dictionary of lists from a list of dictionaries
    record_dict = transpose_dict(records)

    record_node = orm.Dict(dict=record_dict)
    record_node.label = 'delithiation_data'
    record_node.description = 'A node contains the data for the delithiated structures'
    return record_node


def transpose_dict(records):
    """Tranpose a list of dictionary to a dictionary of lists"""
    # Compose a dictionary of lists from a list of dictionaries
    keys = list(records[0].keys())
    record_dict = {key: [] for key in keys}
    for record in records:
        for key in keys:
            record_dict[key].append(record[key])
    return record_dict
