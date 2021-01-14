"""
Bands workchain with a more flexible input
"""
from copy import deepcopy

import numpy as np
import aiida.orm as orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import WorkflowFactory

from aiida_user_addons.process.transform import magnetic_structure_decorate, magnetic_structure_dedecorate
from aiida_user_addons.common.magmapping import create_additional_species


class VaspBandsWorkChain(WorkChain):
    """
    Workchain for running bands calculations.

    This workchain does the following:

    1. Relax the structure if requested (eg. inputs passed to the relax namespace).
    2. Do a SCF singlepoint calculation.
    3. Do a non-scf calculation for bands and dos.

    Inputs must be passed for the SCF calculation, others are optional. The dos calculation
    will only run if the kpoints for DOS are passed or a full specification is given under the
    `dos` input namesace.

    The band structure calculation will run unless `only_dos` is set to `Bool(True)`.

    For magnetic structures, the workchain will internally create additional species for the symmetry
    analysis and regenerate "undecorated" structures with corresponding initial magnetic moments. This
    works for both FM and AFM species. Care should be taken when the MAGMOM is obtained from site projected
    values in case of unexpected symmetry breaking.

    Input for bands and dos calculations are optional. However, if they are needed, the full list of inputs must
    be passed. For the `parameters` node, one may choose to only specify those fields that need to be updated.
    """
    _base_wk_string = 'vaspu.vasp'
    _relax_wk_string = 'vaspu.relax'

    @classmethod
    def define(cls, spec):
        """Initialise the WorkChain class"""
        super().define(spec)
        relax_work = WorkflowFactory(cls._relax_wk_string)
        base_work = WorkflowFactory(cls._base_wk_string)

        spec.input('structure', help='The input structure', valid_type=orm.StructureData)
        spec.input('bands_kpoints',
                   help='Explicit kpoints for the bands. Will not generate kpoints if supplied.',
                   valid_type=orm.KpointsData,
                   required=False)
        spec.input('bands_kpoints_distance',
                   help='Spacing for band distances for automatic kpoints generation.',
                   valid_type=orm.Float,
                   required=False)
        spec.input(
            'dos_kpoints_density',
            help='Kpoints for running DOS calculations in A^-1 * 2pi. Will perform non-SCF DOS calculation is supplied.',
            required=False,
            valid_type=orm.Float,
        )
        spec.expose_inputs(relax_work,
                           namespace='relax',
                           exclude=('structure',),
                           namespace_options={
                               'required': False,
                               'populate_defaults': False,
                               'help': 'Inputs for Relaxation workchain, if needed'
                           })
        spec.expose_inputs(base_work,
                           namespace='scf',
                           exclude=('structure',),
                           namespace_options={
                               'required': True,
                               'populate_defaults': True,
                               'help': 'Inputs for SCF workchain, mandatory'
                           })
        spec.expose_inputs(base_work,
                           namespace='bands',
                           exclude=('structure', 'kpoints'),
                           namespace_options={
                               'required': False,
                               'populate_defaults': False,
                               'help': 'Inputs for bands calculation, if needed'
                           })
        spec.expose_inputs(base_work,
                           namespace='dos',
                           exclude=('structure',),
                           namespace_options={
                               'required': False,
                               'populate_defaults': False,
                               'help': 'Inputs for DOS calculation, if needed'
                           })
        spec.input('clean_children_workdir',
                   valid_type=orm.Str,
                   help='What part of the called children to clean',
                   required=False,
                   default=lambda: orm.Str('none'))
        spec.input('only_dos', required=False, help='Flag for running only DOS calculations')
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.verify_relax,
            ),
            if_(cls.should_run_seekpath)(cls.run_seekpath),
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.verify_scf,
            ),
            cls.run_bands_dos,
            cls.inspect_bands_dos,
        )

        spec.output('primitive_structure', required=False, help='Primitive structure used for band structure calculations')
        spec.output('band_structure', required=False, help='Computed band structure with labels')
        spec.output('seekpath_parameters', help='Parameters used by seekpath', required=False)
        spec.output('dos', required=False)
        spec.output('projectors', required=False)

        spec.exit_code(501, 'ERROR_SUB_PROC_RELAX_FAILED', message='Relaxation workchain failed')
        spec.exit_code(502, 'ERROR_SUB_PROC_SCF_FAILED', message='SCF workchain failed')
        spec.exit_code(503, 'ERROR_SUB_PROC_BANDS_FAILED', message='Band structure workchain failed')
        spec.exit_code(504, 'ERROR_SUB_PROC_DOS_FAILED', message='DOS workchain failed')

    def setup(self):
        """Setup the calculation"""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.bands_kpoints = self.inputs.get('bands_kpoints')
        param = self.inputs.scf.parameters.get_dict()
        if 'magmom' in param['incar'] and not self.inputs.get('only_dos'):
            self.report('Magnetic system passed for BS')
            self.ctx.magmom = param['incar']['magmom']
        else:
            self.ctx.magmom = None

    def should_do_relax(self):
        """Wether we should do relax or not"""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the relaxation"""
        relax_work = WorkflowFactory(self._relax_wk_string)
        inputs = self.exposed_inputs(relax_work, 'relax', agglomerate=True)
        inputs = AttributeDict(inputs)
        inputs.metadata.call_link_label = 'relax'
        inputs.structure = self.ctx.current_structure

        running = self.submit(relax_work, **inputs)
        return self.to_context(workchain_relax=running)

    def verify_relax(self):
        """Verify the relaxation"""
        relax_workchain = self.ctx.workchain_relax
        if not relax_workchain.is_finished_ok:
            self.report('Relaxation finished with Error')
            return self.exit_codes.ERROR_SUB_PROC_RELAX_FAILED

        # Use the relaxed structure as the current structure
        self.ctx.current_structure = relax_workchain.outputs.relax__structure

    def should_run_scf(self):
        """Wether we should run SCF calculation"""
        # TODO - skip if relax gives chgcar and no change in primitive structure
        return True

    def should_run_seekpath(self):
        """
        Seekpath should only run if no explicit bands is provided or we are just
        running for DOS, in which case the original structure is used.
        """
        return 'bands_kpoints' not in self.inputs and (not self.inputs.get('only_dos', False))

    def run_seekpath(self):
        """
        Run seekpath to obtain the primitive structure and bands
        """

        current_structure_backup = self.ctx.current_structure
        inputs = {'reference_distance': self.inputs.get('bands_kpoints_distance', None), 'metadata': {'call_link_label': 'seekpath'}}
        magmom = self.ctx.get('magmom', None)
        # For magnetic structures, create different kinds for the analysis in case that the
        # symmetry should be lowered. This also makes sure that the magnetic moments are consistent
        if magmom:
            decorate_result = magnetic_structure_decorate(self.ctx.current_structure, orm.List(list=magmom))
            decorated = decorate_result['structure']
            # Run seekpath on the decorated structure
            seekpath_results = seekpath_structure_analysis(decorated, **inputs)
            decorated_primitive = seekpath_results['primitive_structure']
            # Convert back to undecorated structures and add consistent magmom input
            dedecorate_result = magnetic_structure_dedecorate(decorated_primitive, decorate_result['mapping'])
            self.ctx.magmom = dedecorate_result['magmom'].get_list()
            self.ctx.current_structure = dedecorate_result['structure']
        else:
            seekpath_results = seekpath_structure_analysis(self.ctx.current_structure, **inputs)
            self.ctx.current_structure = seekpath_results['primitive_structure']

        if not np.allclose(self.ctx.current_structure.cell, current_structure_backup.cell):
            self.report('The primitive structure is not the same as the input structure - using the former for all calculations from now.')
        self.ctx.bands_kpoints = seekpath_results['explicit_kpoints']
        self.out('primitive_structure', self.ctx.current_structure)
        self.out('seekpath_parameters', seekpath_results['parameters'])

    def run_scf(self):
        """
        Run the SCF calculation
        """

        base_work = WorkflowFactory(self._base_wk_string)
        inputs = AttributeDict(self.exposed_inputs(base_work, namespace='scf'))
        inputs.metadata.call_link_label = 'scf'
        inputs.metadata.label = self.inputs.metadata.label + ' SCF'
        inputs.structure = self.ctx.current_structure

        # Turn off cleaning of the working directory
        clean = inputs.get('clean_workdir')
        if clean and clean.value == False:
            pass
        else:
            inputs.clean_workdir = orm.Bool(False)

        # Ensure that writing the CHGCAR file is on
        pdict = inputs.parameters.get_dict()
        if (pdict['incar'].get('lcharg') == False) or (pdict['incar'].get('LCHARG') == False):
            pdict['incar']['lcharg'] = True
            inputs.parameters = orm.Dict(dict=pdict)
            self.report('Correction: setting LCHARG to True')

        # Take magmom from the context, in case that the magmom is rearranged in the primitive cell
        magmom = self.ctx.get('magmom')
        if magmom:
            inputs.parameters = nested_update_dict_node(inputs.parameters, {'incar': {'magmom': magmom}})

        running = self.submit(base_work, **inputs)
        self.report('Running SCF calculation {}'.format(running))
        self.to_context(workchain_scf=running)

    def verify_scf(self):
        """Inspect the SCF calculation"""
        scf_workchain = self.ctx.workchain_scf
        if not scf_workchain.is_finished_ok:
            self.report('SCF workchain finished with Error')
            return self.exit_codes.ERROR_SUB_PROC_SCF_FAILED

        # Store the charge density or remote reference
        if 'chgcar' in scf_workchain.outputs:
            self.ctx.chgcar = scf_workchain.outputs.chgcar
        else:
            self.ctx.chgcar = None
        self.ctx.restart_folder = scf_workchain.outputs.remote_folder
        self.report('SCF calculation {} completed'.format(scf_workchain))

    def run_bands_dos(self):
        """Run the bands and the DOS calculations"""
        base_work = WorkflowFactory(self._base_wk_string)

        # Use the SCF inputs as the base
        inputs = AttributeDict(self.exposed_inputs(base_work, namespace='scf'))
        inputs.structure = self.ctx.current_structure
        inputs.restart_folder = self.ctx.restart_folder

        if self.ctx.chgcar:
            inputs.chgcar = self.ctx.chgcar

        running = {}

        only_dos = self.inputs.get('only_dos')

        if (only_dos is None) or (only_dos.value is False):
            if 'bands' in self.inputs:
                bands_input = AttributeDict(self.exposed_inputs(base_work, namespace='bands'))
            else:
                bands_input = AttributeDict({
                    'settings': orm.Dict(dict={'add_bands': True}),
                    'parameters': orm.Dict(dict={'charge': {
                        'constant_charge': True
                    }}),
                })

            # Special treatment - combine the parameters
            parameters = inputs.parameters.get_dict()
            bands_parameters = bands_input.parameters.get_dict()

            if 'charge' in bands_parameters:
                bands_parameters['charge']['constant_charge'] = True
            else:
                bands_parameters['charge'] = {'constant_charge': True}

            nested_update(parameters, bands_parameters)

            # Apply updated parameters
            inputs.update(bands_input)
            inputs.parameters = orm.Dict(dict=parameters)

            # Check if add_bands
            settings = inputs.get('settings')
            essential = {'parser_settings': {'add_bands': True}}
            if settings is None:
                inputs.settings = orm.Dict(dict=essential)
            else:
                inputs.settings = nested_update_dict_node(settings, essential)

            # Swap with the default kpoints generated
            inputs.kpoints = self.ctx.bands_kpoints

            # Tag the calculation
            inputs.metadata.label = self.inputs.metadata.label + ' BS'
            inputs.metadata.call_link_label = 'bs'

            bands_calc = self.submit(base_work, **inputs)
            running['bands_workchain'] = bands_calc
            self.report('Submitted workchain {} for band structure'.format(bands_calc))

        # Do DOS calculation if dos input namespace is populated or a
        # dos_kpoints input is passed.
        if ('dos_kpoints_density' in self.inputs) or ('dos' in self.inputs):

            if 'dos' in self.inputs:
                dos_input = AttributeDict(self.exposed_inputs(base_work, namespace='dos'))
            else:
                dos_input = AttributeDict({
                    'settings': orm.Dict(dict={'add_dos': True}),
                    'parameters': orm.Dict(dict={'charge': {
                        'constant_charge': True
                    }}),
                })
            # Use the supplied kpoints density for DOS
            if 'dos_kpoints_density' in self.inputs:
                dos_kpoints = orm.KpointsData()
                dos_kpoints.set_cell_from_structure(self.ctx.current_structure)
                dos_kpoints.set_kpoints_mesh_from_density(self.inputs.dos_kpoints_density.value * 2 * np.pi)
                dos_input.kpoints = dos_kpoints

            # Special treatment - combine the parameters
            parameters = inputs.parameters.get_dict()
            dos_parameters = dos_input.parameters.get_dict()
            nested_update(parameters, dos_parameters)

            # Ensure we start from constant charge
            if 'charge' in dos_parameters:
                dos_parameters['charge']['constant_charge'] = True
            else:
                dos_parameters['charge'] = {'constant_charge': True}

            # Apply updated parameters
            inputs.update(dos_input)
            inputs.parameters = orm.Dict(dict=parameters)

            # Check if add_dos
            settings = inputs.get('settings')
            essential = {'parser_settings': {'add_dos': True}}
            if settings is None:
                inputs.settings = orm.Dict(dict=essential)
            else:
                inputs.settings = nested_update_dict_node(settings, essential)

            # Set the label
            inputs.metadata.label = self.inputs.metadata.label + ' DOS'
            inputs.metadata.call_link_label = 'dos'

            dos_calc = self.submit(base_work, **inputs)
            running['dos_workchain'] = dos_calc
            self.report('Submitted workchain {} for DOS'.format(dos_calc))

        return self.to_context(**running)

    def inspect_bands_dos(self):
        """Inspect the bands and dos calculations"""

        exit_code = None

        if 'bands_workchain' in self.ctx:
            bands = self.ctx.bands_workchain
            if not bands.is_finished_ok:
                self.report('Bands calculation finished with error, exit_status: {}'.format(bands, bands.exit_status))
                exit_code = self.exit_codes.ERROR_SUB_PROC_BANDS_FAILED
            self.out('band_structure', compose_labelled_bands(bands.outputs.bands, bands.inputs.kpoints))
        else:
            bands = None

        if 'dos_workchain' in self.ctx:
            dos = self.ctx.dos_workchain
            if not dos.is_finished_ok:
                self.report('DOS calculation finished with error, exit_status: {}'.format(dos.exit_status))
                exit_code = self.exit_codes.ERROR_SUB_PROC_DOS_FAILED

            # Attach outputs
            self.out('dos', dos.outputs.dos)
            if 'projectors' in dos.outputs:
                self.out('projectors', dos.outputs.projectors)
        else:
            dos = None

        return exit_code

    def on_terminated(self):
        """
        Clean the remote directories of all called childrens
        """

        super(VaspBandsWorkChain, self).on_terminated()

        if self.inputs.clean_children_workdir.value != 'none':
            cleaned_calcs = []
            for called_descendant in self.node.called_descendants:
                if isinstance(called_descendant, orm.CalcJobNode):
                    try:
                        called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                        cleaned_calcs.append(called_descendant.pk)
                    except (IOError, OSError, KeyError):
                        pass

            if cleaned_calcs:
                self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))


def nested_update(dict_in, update_dict):
    """Update the dictionary - combine nested subdictionary with update as well"""
    for key, value in update_dict.items():
        if key in dict_in and isinstance(value, (dict, AttributeDict)):
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


@calcfunction
def seekpath_structure_analysis(structure, **kwargs):
    """Primitivize the structure with SeeKpath and generate the high symmetry k-point path through its Brillouin zone.
    This calcfunction will take a structure and pass it through SeeKpath to get the normalized primitive cell and the
    path of high symmetry k-points through its Brillouin zone. Note that the returned primitive cell may differ from the
    original structure in which case the k-points are only congruent with the primitive cell.
    The keyword arguments can be used to specify various Seekpath parameters, such as:
        with_time_reversal: True
        reference_distance: 0.025
        recipe: 'hpkot'
        threshold: 1e-07
        symprec: 1e-05
        angle_tolerance: -1.0
    Note that exact parameters that are available and their defaults will depend on your Seekpath version.
    """
    from aiida.tools import get_explicit_kpoints_path

    # All keyword arugments should be `Data` node instances of base type and so should have the `.value` attribute
    unwrapped_kwargs = {key: node.value for key, node in kwargs.items() if isinstance(node, orm.Data)}

    return get_explicit_kpoints_path(structure, **unwrapped_kwargs)


@calcfunction
def compose_labelled_bands(bands, kpoints):
    """
    Add additional information from the kpoints allow richer informations
    to be stored such as band structure labels.
    """
    new_bands = deepcopy(bands)
    new_bands.set_kpointsdata(kpoints)
    return new_bands
