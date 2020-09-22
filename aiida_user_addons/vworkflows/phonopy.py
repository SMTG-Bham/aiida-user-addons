"""
Workchain implementation for performing phonon calculation using `aiida-phonopy`

Difference to the workflow in `aiida-phonopy`: here we do not use and *immigrant* method
for supporting imported calculations. Also, add the relaxation step to fully converge the
structure first - potentially allowing a direct structure -> phonon property workflow.

The a few VASP specific points has been marked. In theory, the work chain can be adapted to
support any code that does force and energy output.
"""
from aiida.engine import WorkChain, if_, ToContext
import aiida.orm as orm

from aiida.plugins import WorkflowFactory
from ..common.opthold import OptionHolder, typed_field, required_field
from aiida_phonopy.common.utils import (
    get_force_constants,
    get_nac_params,
    get_phonon,
    generate_phonopy_cells,
    check_imported_supercell_structure,
    from_node_id_to_aiida_node_id,
    get_data_from_node_id,
    get_vasp_force_sets_dict,
)


class VaspAutoPhononWorkChain(WorkChain):
    _relax_entrypoint = 'vaspu.relax'
    _relax_chain = WorkflowFactory(_relax_entrypoint)
    _singlepoint_entrypoint = 'vaspu.vasp'
    _singlepoint_chain = WorkflowFactory(_singlepoint_entrypoint)

    @classmethod
    def define(cls, spec):

        super().define(spec)
        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(cls.run_relaxation, cls.inspect_relaxation),
            cls.create_displacements,
            cls.run_force_and_nac_calcs,
            cls.create_force_set_and_constants,
            if_(cls.remote_phonopy)(cls.run_phonopy_remote, cls.collect_remote_run_data).else_(
                cls.create_force_constants,
                cls.run_phonopy_local,
            ),
        )

        # Standard calculation inputs
        spec.expose_inputs(cls._relax_chain,
                           namespace='relax',
                           exclude=('structure',),
                           namespace_options={
                               'required': False,
                               'help': 'Inputs for the relaxation to be performed.',
                               'populate_defaults': False,
                           })
        spec.expose_inputs(cls._singlepoint_chain,
                           namespace='singlepoint',
                           exclude=('structure',),
                           namespace_options={
                               'required': True,
                               'help': 'Additional inputs for the singlepoint calculations.',
                               'populate_defaults': True,
                           })
        spec.expose_inputs(cls._singlepoint_chain,
                           namespace='nac',
                           exclude=('structure',),
                           namespace_options={
                               'required': False,
                               'populate_defaults': False,
                               'help': 'Inputs for the DFPT NAC calculation.'
                           })
        # Phonon specific inputs
        spec.input('remote_phonopy', default=lambda: orm.Bool(False), help='Run phonopy as a remote code.')
        spec.input('symmetry_tolerance', valid_type=orm.Float, default=lambda: orm.Float(1e-5))
        spec.input('subtract_residual_forces', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('structure', valid_type=orm.StructureData, help='Structure of which the phonons should calculated')
        spec.input('phonon_settings',
                   valid_type=orm.Dict,
                   validator=PhononSettings.validate_dict,
                   help='Settings for the underlying phonopy calculations')
        spec.input('phonon_code', valid_type=orm.Code, help='Code for the phonopy for remote calculations', required=False)
        spec.input('options', valid_type=orm.Dict, help='Options for the remote phonopy calculation', required=False)

        # Phonon specific outputs
        spec.output('force_constants', valid_type=orm.ArrayData, required=False)
        spec.output('primitive', valid_type=orm.StructureData, required=False)
        spec.output('supercell', valid_type=orm.StructureData, required=False)
        spec.output('force_sets', valid_type=orm.ArrayData, required=False)
        spec.output('nac_params', valid_type=orm.ArrayData, required=False)
        spec.output('thermal_properties', valid_type=orm.XyData, required=False)
        spec.output('band_structure', valid_type=orm.BandsData, required=False)
        spec.output('dos', valid_type=orm.XyData, required=False)
        spec.output('pdos', valid_type=orm.XyData, required=False)
        spec.output('phonon_setting_info', valid_type=orm.Dict, required=True)
        spec.output('relaxed_structure',
                    valid_type=orm.StructureData,
                    required=False,
                    help='The output structure of the high precision relaxation, used for phonon calculations.')

        spec.exit_code(501, 'ERROR_RELAX_FAILURE', message='Initial relaxation has failed!')

    def setup(self):
        """Setup the workspace"""
        # Current structure for calculations
        self.ctx.current_structure = self.inputs.structure
        self.ctx.label = self.inputs.metadata.get('label', '')

    def should_run_relax(self):
        if 'relax' in self.inputs:
            return True
        else:
            self.report('Not performing relaxation - assuming the input structure is fully relaxed.')

    def run_relaxation(self):
        """Perform high-precision relaxation of the initial structure"""

        inputs = self.exposed_inputs(self._relax_chain, 'relax')
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'high_prec_relax'
        if 'label' not in inputs.metadata or (not inputs.metadata.label):
            inputs.metadata.label = self.ctx.label + ' HIGH-PREC RELAX'

        running = self.submit(self._relax_chain, **inputs)

        self.report(f'Submitted high-precision relaxation {running}')
        return ToContext(relax_calc=running)

    def inspect_relaxation(self):
        """Check if the relaxation finished OK"""
        if 'relax_calc' not in self.ctx:
            raise RuntimeError('Relaxation workchain not found in the context')

        workchain = self.ctx.relax_calc
        if not workchain.is_finished_ok:
            self.report('Relaxation finished with error, abort further actions')
            return self.exit_codes.ERROR_RELAX_FAILURE  # pylint: disable=no-member

        # All OK
        self.ctx.current_structure = workchain.outputs.relax__structure  # NOTE: this is workchain specific
        self.report('Relaxation finished OK, recorded the relaxed structure')
        self.out('relaxed_structure', self.ctx.current_structure)

    def create_displacements(self):
        """Create displacements using phonopy"""

        self.report('Creating displacements')
        phonon_settings = self.inputs.phonon_settings.get_dict()

        # Check if we are doing magnetic calculations
        force_calc_inputs = self.exposed_inputs(self._singlepoint_chain, 'singlepoint')
        relax_calc_inputs = self.exposed_inputs(self._relax_chain, 'relax')

        # Fetch the magmom from the relaxation calculation (eg. for the starting structure)
        try:
            magmom = relax_calc_inputs.vasp.parameters['vasp'].get('magmom')
        except AttributeError:
            magmom = None

        # MAGMOM tag in the phonon_settings input port take the precedence
        if magmom and ('magmom' not in phonon_settings):
            self.report('Using MAGMOM from the inputs for the relaxation calculations')
            phonon_settings['magmom'] = magmom
            phonon_settings_dict = orm.Dict(dict=phonon_settings)
        else:
            phonon_settings_dict = self.inputs.phonon_settings

        if 'supercell_matrix' not in phonon_settings:
            raise RuntimeError("Must supply 'supercell_matrix' in the phonon_settings input.")

        kwargs = {}
        return_vals = generate_phonopy_cells(phonon_settings_dict, self.ctx.current_structure, self.inputs.symmetry_tolerance, **kwargs)

        # Store these in the context and set the output
        for key in ('phonon_setting_info', 'primitive', 'supercell'):
            self.ctx[key] = return_vals[key]
            self.out(key, self.ctx[key])

        self.ctx.supercell_structures = {}

        for key in return_vals:
            if 'supercell_' in key:
                self.ctx.supercell_structures[key] = return_vals[key]

        if self.inputs.subtract_residual_forces:
            # The 000 structure is the original supercell
            digits = len(str(len(self.ctx.supercell_structures)))
            label = 'supercell_{}'.format('0'.zfill(digits))
            self.ctx.supercell_structures[label] = return_vals['supercell']

        self.report('Supercells for phonon calculations created')

    def run_force_and_nac_calcs(self):
        """Submit the force and non-analytical correction calculations"""
        # Forces
        force_calc_inputs = self.exposed_inputs(self._singlepoint_chain, 'singlepoint')

        magmom = self.ctx.phonon_setting_info.get_dict().get('_supercell_magmom')
        if magmom:
            self.report('Using MAGMOM from the phonopy output')
            param = force_calc_inputs.parameters.get_dict()
            param['vasp']['magmom'] = magmom
            force_calc_inputs.parameters = orm.Dict(dict=param)

        # Ensure we parser the forces
        ensure_parse_objs(force_calc_inputs, ['forces'])

        for key, node in self.ctx.supercell_structures.items():
            label = 'force_calc_' + key.split('_')[-1]
            force_calc_inputs.structure = node
            force_calc_inputs.metadata.call_link_label = label
            if 'label' not in force_calc_inputs.metadata or (not force_calc_inputs.metadata.label):
                force_calc_inputs.metadata.label = self.ctx.label + ' FC_' + key.split('_')[-1]

            running = self.submit(self._singlepoint_chain, **force_calc_inputs)

            self.report('Submitted {} for {}'.format(running, label))
            self.to_context(**{label: running})

        if self.is_nac():
            self.report('calculate born charges and dielectric constant')
            nac_inputs = self.exposed_inputs(self._singlepoint_chain, 'nac')
            # NAC needs to use the primitive structure!
            nac_inputs.structure = self.ctx.primitive
            nac_inputs.metadata.call_link_label = 'nac_calc'
            if 'label' not in nac_inputs.metadata or (not nac_inputs.metadata.label):
                nac_inputs.metadata.label = self.ctx.label + ' NAC'
            ensure_parse_objs(nac_inputs, ['dielectrics', 'born_charges'])

            running = self.submit(self._singlepoint_chain, **nac_inputs)
            self.report('Submissted calculation for nac: {}'.format(running))
            self.to_context(**{'born_and_epsilon_calc': running})

        return

    def create_force_set_and_constants(self):
        """Create the force set and constants from the finished calculations"""

        self.report('Creating force set and nac (if applicable)')
        forces_dict = collect_vasp_forces_and_energies(self.ctx, self.ctx.supercell_structures, 'force_calc')

        # Will set force_sets, supercell_forces, supercell_energy - the latter two are optional
        for key, value in get_vasp_force_sets_dict(**forces_dict).items():
            self.ctx[key] = value
            self.out(key, self.ctx[key])

        if self.is_nac():

            self.report('Create nac data')
            calc = self.ctx.born_and_epsilon_calc
            # NOTE: this is VASP specific outputs -- but I can implement the same for CASTEP plugin
            if isinstance(calc, dict):  # For imported calculations - not used here
                calc_dict = calc
                structure = calc['structure']
            else:
                calc_dict = calc.outputs
                structure = calc.inputs.structure

            if 'born_charges' not in calc_dict:
                raise RuntimeError('Born effective charges could not be found ' 'in the calculation. Please check the calculation setting.')
            if 'dielectrics' not in calc_dict:
                raise RuntimeError('Dielectric constant could not be found ' 'in the calculation. Please check the calculation setting.')

            self.ctx.nac_params = get_nac_params(calc_dict['born_charges'], calc_dict['dielectrics'], structure,
                                                 self.inputs.symmetry_tolerance)
            self.out('nac_params', self.ctx.nac_params)

    def run_phonopy_remote(self):
        """Run phonopy as remote code"""
        self.report('run remote phonopy calculation')

        code_string = self.inputs.code_string.value
        builder = orm.Code.get_from_string(code_string).get_builder()
        builder.structure = self.ctx.current_structure
        builder.settings = self.ctx.phonon_setting_info  # This was generated by the earlier call
        builder.metadata.options.update(self.inputs.options)
        builder.metadata.label = self.ctx.label
        builder.force_sets = self.ctx.force_sets  # Generated earlier
        if 'nac_params' in self.ctx:
            builder.nac_params = self.ctx.nac_params
            builder.primitive = self.ctx.primitive
        future = self.submit(builder)

        self.report('Submitted phonopy calculation: {}'.format(future.pk))
        self.to_context(**{'phonon_properties': future})

    def create_force_constants(self):
        self.report('Creating force constants')

        self.ctx.force_constants = get_force_constants(self.ctx.current_structure, self.ctx.phonon_setting_info, self.ctx.force_sets)
        self.out('force_constants', self.ctx.force_constants)

    def run_phonopy_local(self):
        """
        Run phonopy in the local interpreter.
        WARRNING! This could put heavy strain on the local python process and
        potentially affect the daemon worker executions. Long running time
        can make the work lose contact with the daemon and give rise to double
        execution problems. USE WITH CAUTION
        """
        self.report('Perform phonopy calculation in workchain')

        params = {}
        if 'nac_params' in self.ctx:
            params['nac_params'] = self.ctx.nac_params
        result = get_phonon(self.inputs.structure, self.ctx.phonon_setting_info, self.ctx.force_constants, **params)
        self.out('thermal_properties', result['thermal_properties'])
        self.out('dos', result['dos'])
        self.out('band_structure', result['band_structure'])

        self.report('Completed local phonopy calculation, workchain finished.')

    def collect_remote_run_data(self):
        """Collect the data from a remote phonopy run"""
        self.report('Collecting  data from a remote phonopy run')
        ph_props = ('thermal_properties', 'dos', 'pdos', 'band_structure', 'force_constants')

        for prop in ph_props:
            if prop in self.ctx.phonon_properties.outputs:
                self.out(prop, self.ctx.phonon_properties.outputs[prop])

        self.report('Completed collecting remote phonopy data, workchain finished.')

    def is_nac(self):
        """
        Check if nac calculations should be performed.
        Returns trun if the 'nac' input namespace exists.
        """
        return bool(self.inputs.get('nac'))

    def remote_phonopy(self):
        """Weither to run phonopy as a remote code"""
        node = self.inputs.remote_phonopy
        return bool(node)


def collect_vasp_forces_and_energies(ctx, ctx_supercells, prefix='force_calc'):
    """
    Collect forces and energies from VASP calculations.
    This is essentially for pre-process before dispatching to the calcfunction for creating
    the force_set

    Returns:
        A dictionary with keys like "forces_<num>" and "misc_<num>", mapping to aiida nodes
    """
    forces_dict = {}
    for key in ctx_supercells:
        # key: e.g. "supercell_001", "phonon_supercell_001"
        num = key.split('_')[-1]  # e.g. "001"
        calc = ctx['{}_{}'.format(prefix, num)]

        # Also works for imported calculations
        if type(calc) is dict:
            calc_dict = calc
        else:
            calc_dict = calc.outputs
        if ('forces' in calc_dict and 'final' in calc_dict['forces'].get_arraynames()):
            forces_dict['forces_{}'.format(num)] = calc_dict['forces']
        else:
            raise RuntimeError('Forces could not be found in calculation {}.'.format(num))

        if ('misc' in calc_dict and
                'total_energies' in calc_dict['misc'].keys()):  # needs .keys() - calc_dict can be a dict or a LinkManager
            forces_dict['misc_{}'.format(num)] = calc_dict['misc']

    return forces_dict


class PhononSettings(OptionHolder):
    """Options for phonon_settings input"""
    _allowed_options = ('supercell_matrix', 'mesh', 'distance', 'primitive_matrix')
    supercell_matrix = required_field('supercell_matrix', (list,), 'Supercell matrix for phonons')
    primitive_matrix = typed_field('primitive_matrix', (list, str), 'primitive matrix for phonons', 'auto')
    mesh = required_field('mesh', (int,), 'Mesh for phonon calculation')
    magmom = typed_field('magmom', (list,), 'Starting magnetic moments', None)
    distance = typed_field('distance', (
        float,
        int,
    ), 'Distance for band structure', None)


def nested_update(dict_in, update_dict):
    """Update the dictionary - combine nested subdictionary with update as well"""
    for key, value in update_dict.items():
        if key in dict_in and isinstance(value, (dict, orm.AttributeDict)):
            nested_update(dict_in[key], value)
        else:
            dict_in[key] = value
    return dict_in


def nested_update_dict_node(dict_node, update_dict):
    """Utility to update a Dict node in a nested way"""
    pydict = dict_node.get_dict()
    nested_update(pydict, update_dict)
    if pydict == dict_node.get_dict():
        return dict_node
    else:
        return orm.Dict(dict=pydict)


def ensure_parse_objs(input_port, objs):
    """
    Ensure parser will parse certain objects

    Arguments:
        input_port: input port to be update, assume the existence of `settings`
        objs: a list of objects to include, for example ['structure', 'forces']

    Returns:
        process_port: the port with the new settings
    """
    update = {'parser_settings': {'add_{}'.format(obj): True for obj in objs}}
    if 'settings' not in input_port:
        input_port.settings = orm.Dict(dict=update)
    else:
        settings = input_port.settings
        nested_update_dict_node(settings, update)
        input_port.settings = settings
    return input_port
