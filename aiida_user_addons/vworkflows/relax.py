"""
Relaxation workchain for VASP

Unfortunately, VASP does not check the convergence criteria properly:
- it only check *either* force or energy convergence between the last two iterations
- relaxation is performed with *constant basis set*, so a final singlepoint calculation is necessary if cell is to be relaxed
- no check for the convergence of the atomic positions
- no check about the convergence of the cell volume

Hence we have to control it externally to make sure the structures are properly relaxed,
and perform additional singlepoint calculation where necessary.

In addition, using this workchain, there is no need to set the IBRION, NSW, ISIF and ALGO tags explicitly,
the action can be controlled using the `relax_setting` input port. This will be merged with the `parameters` and
passed downstream to the `VaspBaseWorkChain` which will work out the correct combinations for the three.

CHANGELOG

0.2.1 - added `hybrid_calc_bootstrap` options
0.3.0 - make singpoint calculation reuse the `restart_folder`
0.3.1 - called processes now have link labels. Added customisable `settings` and `options` for the final relaxation.


"""
# pylint: disable=attribute-defined-outside-init
import numpy as np

import aiida.orm as orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.engine import WorkChain, append_, while_, if_, ToContext
from aiida.plugins import WorkflowFactory
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_vasp.utils.aiida_utils import get_data_class
from aiida_vasp.utils.workchains import compose_exit_code

from .mixins import WithVaspInputSet
from ..common.opthold import OptionHolder, typed_field
from .common import OVERRIDE_NAMESPACE

__version__ = '0.4.0'

# Change log
# 0.4.0 update such `vasp` namespace in `parameters` is renamed to `incar`


class VaspRelaxWorkChain(WorkChain, WithVaspInputSet):
    """Structure relaxation workchain."""
    _verbose = True
    _base_workchain_string = 'vaspu.vasp'
    _base_workchain = WorkflowFactory(_base_workchain_string)

    @classmethod
    def define(cls, spec):
        super(VaspRelaxWorkChain, cls).define(spec)
        spec.expose_inputs(cls._base_workchain, 'vasp', exclude=('structure',))
        spec.input('structure', valid_type=(get_data_class('structure'), get_data_class('cif')))
        spec.input('static_calc_parameters',
                   valid_type=get_data_class('dict'),
                   required=False,
                   serializer=to_aiida_type,
                   help="""
                   The parameters (INCAR) to be used in the final static calculation.
                   """)
        spec.input('static_calc_settings',
                   valid_type=get_data_class('dict'),
                   required=False,
                   serializer=to_aiida_type,
                   help="""
                   The full settings Dict to be used in the final static calculation.
                   """)
        spec.input('static_calc_options',
                   valid_type=get_data_class('dict'),
                   required=False,
                   serializer=to_aiida_type,
                   help="""
                   The full options Dict to be used in the final static calculation.
                   """)
        spec.input('relax_settings',
                   valid_type=get_data_class('dict'),
                   validator=RelaxOptions.validate_dict,
                   serializer=to_aiida_type,
                   help=RelaxOptions.get_description())
        spec.input('verbose', required=False, help='Increased verbosity.', valid_type=orm.Bool, serializer=to_aiida_type)
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        spec.exit_code(300,
                       'ERROR_MISSING_REQUIRED_OUTPUT',
                       message='the called workchain does not contain the necessary relaxed output structure')
        spec.exit_code(420, 'ERROR_NO_CALLED_WORKCHAIN', message='no called workchain detected')
        spec.exit_code(500, 'ERROR_UNKNOWN', message='unknown error detected in the relax workchain')
        spec.exit_code(502, 'ERROR_OVERRIDE_PARAMETERS', message='there was an error overriding the parameters')
        spec.exit_code(
            600,
            'ERROR_RELAX_NOT_CONVERGED',
            message='Ionic relaxation was not converged after the maximum number of iterations has been spent',
        )
        spec.exit_code(601,
                       'ERROR_FINAL_SCF_HAS_RESIDUAL_FORCE',
                       message=('The final singlepoint calculation has increased residual forces. '
                                'This may be caused by electronic solver converging to a different solution. '
                                'Care should be taken to investigate the results.'))
        spec.outline(
            cls.initialize,
            if_(cls.perform_relaxation)(
                while_(cls.run_next_relax)(
                    cls.run_relax,
                    cls.verify_last_relax,
                    cls.analyze_convergence,
                ),
                cls.store_relaxed,
            ),
            cls.init_relaxed,
            if_(cls.should_run_static_calculation)(
                cls.run_static_calculation,
                cls.verify_next_workchain,
                ),
            cls.results,
            cls.finalize
        )  # yapf: disable

        spec.expose_outputs(cls._base_workchain)
        spec.output('relax.structure', valid_type=get_data_class('structure'), required=False)

    def initialize(self):
        """Initialize."""

        # Initialise the contexts
        self.ctx.exit_code = self.exit_codes.ERROR_UNKNOWN  # pylint: disable=no-member
        self.ctx.is_converged = False
        self.ctx.relax = False
        self.ctx.iteration = 0
        self.ctx.workchains = []
        self.ctx.inputs = AttributeDict()  # This may not be necessary anymore
        self.ctx.relax_settings = AttributeDict(self.inputs.relax_settings.get_dict())  # relax_settings controls the logic of the workchain

        # Storage space for the inputs
        self.ctx.relax_input_additions = self._init_relax_input_additions()
        self.ctx.static_input_additions = AttributeDict()

        # Set the verbose flag
        if 'verbose' in self.inputs:
            self.ctx.verbose = self.inputs.verbose.value
        else:
            self.ctx.verbose = self._verbose  # Hard-coded default

        # Take the input structure as the "current" structure
        self.ctx.current_structure = self.inputs.structure

        # Make sure we parse the output structure when we want to perform
        # relaxations (override if contrary entry exists).
        if 'settings' in self.inputs.vasp:
            settings = self.inputs.vasp.settings
        else:
            settings = orm.Dict(dict={})

        if self.perform_relaxation():
            settings = nested_update_dict_node(settings, {'parser_settings': {
                'add_structure': True,
                'add_trajectory': True,
            }})

        # Update the settings for the relaxation
        if settings.get_dict():
            self.ctx.relax_input_additions.settings = settings

        # Hybrid calculation boot-strapping
        if self.ctx.relax_settings.get('hybrid_calc_bootstrap'):
            self.ctx.hybrid_status = 'dft'
            if not self.ctx.relax_settings.get('reuse'):
                self.report('Enable `reuse` mode because hybrid calculation bootstrapping has been requested.')
                self.ctx.relax_settings['reuse'] = True
        else:
            self.ctx.hybrid_status = None

        # If we are reusing existing calculations, make sure the remote folders are not cleaned
        if self.ctx.relax_settings.get('reuse'):
            clean_tmp = self.inputs.vasp.get('clean_workdir')
            if clean_tmp and clean_tmp.value:
                self.report('Disable clean_workdir for downstream workflows since `reuse` is requested.')
                self.ctx.relax_input_additions['clean_workdir'] = orm.Bool(False)

    def _init_relax_input_additions(self):
        """
        Initialise the `relax_additions` field inside the context.
        It is a AttributeDict that contains the inputs that should be updated
        while performing the relaxation.
        """

        additions = AttributeDict()
        # Update the "relax" field inside the parameters - this is needed because some of the
        # settings will be translated into VASP parameters
        if self.perform_relaxation():
            parameters = nested_update_dict_node(self.inputs.vasp.parameters, {'relax': self.ctx.relax_settings})
            additions.parameters = parameters

        return additions

    def run_next_relax(self):
        within_max_iterations = bool(self.ctx.iteration < self.ctx.relax_settings.convergence_max_iterations)
        return bool(within_max_iterations and not self.ctx.is_converged)

    def init_relaxed(self):
        """Initialize a calculation based on a relaxed or assumed relaxed structure."""

        # Did not perform the relaxation - going into the final singlepoint directly
        if not self.perform_relaxation():
            if self.is_verbose():
                self.report('skipping structure relaxation and forwarding input to the next workchain.')
        else:
            # For the final static run we do not need to parse the output structure
            if 'settings' in self.inputs.vasp:
                self.ctx.static_input_additions.settings = nested_update_dict_node(
                    self.inputs.vasp.settings, {'parser_settings': {
                        'add_structure': False,
                        'add_trajectory': False,
                    }})

            # Apply overrides if supplied
            if 'static_calc_settings' in self.inputs:
                self.ctx.static_input_additions.settings = self.inputs.static_calc_settings

            if 'static_calc_options' in self.inputs:
                self.ctx.static_input_additions.options = self.inputs.static_calc_options

            # Override INCARs for the final relaxation
            if 'static_calc_parameters' in self.inputs:
                self.ctx.static_input_additions.parameters = nested_update_dict_node(self.inputs.vasp.parameters,
                                                                                     self.inputs.static_parameters.get_dict())
            if self.is_verbose():
                self.report('performing a final calculation using the relaxed structure.')

    def run_relax(self):
        """Perform the relaxation"""
        self.ctx.iteration += 1

        inputs = self.exposed_inputs(self._base_workchain, 'vasp')
        inputs.structure = self.ctx.current_structure

        # Attach previous calculation's folder if requested
        if self.ctx.relax_settings.get('reuse', False):
            restart_folder = self.ctx.get('current_restart_folder')  # There might not be any yet
            if restart_folder:
                if self.ctx.get('verbose'):
                    self.report('Using previous remote folder <{}> for restart'.format(restart_folder))
                inputs.restart_folder = restart_folder

        # Update the input with whatever stored in the relax_input_additions attribute dict
        inputs.update(self.ctx.relax_input_additions)
        if 'label' not in inputs.metadata:
            inputs.metadata.label = self.inputs.metadata.get('label', '')

        # Check if we need to boot strap hybrid calculation
        if self.ctx.get('hybrid_status') == 'dft':
            # Turn off HF and turn off relaxation
            inputs.parameters = nested_update_dict_node(
                inputs.parameters,
                {
                    OVERRIDE_NAMESPACE: {
                        'lhfcalc': False,
                        'isym': 2,  # Standard DFT needs ISYM=2
                    },
                    'relax': {
                        # This turns off any relaxation
                        'positions': False,
                        'volume': False,
                        'shape': False,
                    }
                })
            # Decrease the iteration number
            self.ctx.iteration -= 1
            # Update the wallclock seconds
            wallclock = self.ctx.relax_settings.get('hybrid_calc_bootstrap_wallclock')
            if wallclock:
                inputs.options = nested_update_dict_node(inputs.options, {'max_wallclock_seconds': wallclock})

        # Label the calculation and links by iteration number
        inputs.metadata.label += ' ITER {:02d}'.format(self.ctx.iteration)
        inputs.metadata.call_link_label = 'relax_{:02d}'.format(self.ctx.iteration)

        running = self.submit(self._base_workchain, **inputs)
        self.report('launching {}<{}> iterations #{}'.format(self._base_workchain.__name__, running.pk, self.ctx.iteration))

        return ToContext(workchains=append_(running))

    def run_static_calculation(self):
        """Perform the relaxation"""
        self.ctx.iteration += 1

        inputs = self.exposed_inputs(self._base_workchain, 'vasp')
        inputs.structure = self.ctx.current_structure

        # Attach previous calculation's folder if requested
        if self.ctx.relax_settings.get('reuse', False):
            restart_folder = self.ctx.get('current_restart_folder')  # There might not be any yet
            if restart_folder:
                if self.ctx.get('verbose'):
                    self.report('Using previous remote folder <{}> for restart'.format(restart_folder))
                inputs.restart_folder = restart_folder

        if 'label' not in inputs.metadata:
            inputs.metadata.label = self.inputs.metadata.get('label', '') + ' SP'

        # Label the calculation and links by iteration number
        inputs.metadata.call_link_label = 'singlepoint'

        # Update the input with whatever stored in the relax_input_additions attribute dict
        inputs.update(self.ctx.static_input_additions)

        running = self.submit(self._base_workchain, **inputs)
        self.report('launching {}<{}> iterations #{}'.format(self._base_workchain.__name__, running.pk, self.ctx.iteration))
        return ToContext(workchains=append_(running))

    def verify_next_workchain(self):
        """Verify and inherit exit status from child workchains."""

        try:
            workchain = self.ctx.workchains[-1]
        except IndexError:
            self.report('There is no {} in the called workchain list.'.format(self._base_workchain.__name__))
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member

        # Inherit exit status from last workchain (supposed to be
        # successfull)
        next_workchain_exit_status = workchain.exit_status
        next_workchain_exit_message = workchain.exit_message
        if not next_workchain_exit_status:
            self.ctx.exit_code = self.exit_codes.NO_ERROR  # pylint: disable=no-member
        else:
            self.ctx.exit_code = compose_exit_code(next_workchain_exit_status, next_workchain_exit_message)
            self.report('The called {}<{}> returned a non-zero exit status. '
                        'The exit status {} is inherited'.format(workchain.__class__.__name__, workchain.pk, self.ctx.exit_code))

        return self.ctx.exit_code

    def verify_last_relax(self):
        """Verify and inherit exit status from the last relaxation"""

        try:
            workchain = self.ctx.workchains[-1]
        except IndexError:
            self.report('There is no {} in the called workchain list.'.format(self._base_workchain.__name__))
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member

        # Inherit exit status from last workchain (supposed to be
        # successfull)
        next_workchain_exit_status = workchain.exit_status
        next_workchain_exit_message = workchain.exit_message
        if not next_workchain_exit_status:
            self.ctx.exit_code = self.exit_codes.NO_ERROR  # pylint: disable=no-member
        elif 'misc' in workchain.outputs and 'structure' in workchain.outputs and 'maximum_force' in workchain.outputs.misc.get_dict():
            self.ctx.exit_code = self.exit_codes.NO_ERROR  # pylint: disable=no-member
            self.report('The called {}<{}> returned a non-zero exit status. '
                        'Continue the workflow as "misc" and "structure" outputs are present.'.format(
                            workchain.__class__.__name__, workchain.pk))
        else:
            self.ctx.exit_code = compose_exit_code(next_workchain_exit_status, next_workchain_exit_message)
            self.report('The called {}<{}> returned a non-zero exit status. '
                        'The exit status {} is inherited'.format(workchain.__class__.__name__, workchain.pk, self.ctx.exit_code))

        return self.ctx.exit_code

    def analyze_convergence(self):
        """
        Analyze the convergence of the relaxation.

        Compare the input and output structures of the most recent relaxation run. If volume,
        shape and ion positions are all within a given threshold, consider the relaxation converged.
        """
        workchain = self.ctx.workchains[-1]
        # Double check presence of structure
        if 'structure' not in workchain.outputs:
            self.report('The {}<{}> for the relaxation run did not have an '
                        'output structure and most likely failed. However, '
                        'its exit status was zero.'.format(workchain.__class__.__name__, workchain.pk))
            return self.exit_codes.ERROR_MISSING_REQUIRED_OUTPUT  # pylint: disable=no-member

        if self.ctx.hybrid_status == 'dft':
            self.report('Competed initial DFT calculation - skipping convergence checks and process to hybrid calculation.')
            self.ctx.hybrid_status = 'hybrid'
            self.ctx.current_restart_folder = workchain.outputs.remote_folder
            return self.exit_codes.NO_ERROR  # pylint: disable=no-member

        self.ctx.previous_structure = self.ctx.current_structure
        self.ctx.current_structure = workchain.outputs.structure

        conv_mode = self.ctx.relax_settings.convergence_mode
        # Assign the two structure used for comparison
        if conv_mode == 'inout':
            compare_from = self.ctx.previous_structure
            compare_to = self.ctx.current_structure
        elif conv_mode == 'last':
            traj = workchain.outputs.trajectory
            if traj.numsteps > 1:
                compare_from = get_step_structure(workchain.outputs.trajectory, -2)  # take take second last structure
                compare_to = get_step_structure(workchain.outputs.trajectory, -1)  # take take second last structure
            else:
                self.report('Warning - no enough number of steps to compare - using input/output structures instead.')
                compare_from = self.ctx.previous_structure
                compare_to = self.ctx.current_structure
        else:
            raise RuntimeError('Convergence mode {} is not a valid option'.format(conv_mode))

        converged = True
        relax_settings = self.ctx.relax_settings
        if relax_settings.convergence_on:
            if self.is_verbose():
                self.report('Checking the convergence of the relaxation.')
            comparison = compare_structures(compare_from, compare_to)
            delta = comparison.absolute if relax_settings.convergence_absolute else comparison.relative
            if relax_settings.positions:
                # For positions it only makes sense to check the absolute change
                converged &= self.check_positions_convergence(comparison.absolute)
            if relax_settings.volume:
                converged &= self.check_volume_convergence(delta)
            if relax_settings.shape:
                converged &= self.check_shape_convergence(delta)

            # BONAN: Check force - this is becuase the underlying VASP calculation may not have finished with
            # fully converge geometry, and the vasp plugin does not check it.
            force_cut_off = relax_settings.get('force_cutoff')
            max_force = workchain.outputs.misc.get_attribute('maximum_force')
            if force_cut_off is not None and max_force > force_cut_off:
                self.report(f'Maximum force in the structure {max_force:.4g} excess the cut-off limit {force_cut_off:.4g} - NOT OK')
                converged = False
            elif self.is_verbose():
                self.report(f'Maximum force in the structure {max_force:.4g} - OK')

            if not converged:
                if self.ctx.get('verbose', self._verbose):
                    self.report('Relaxation did not converge, restarting the relaxation.')
            else:
                if self.is_verbose():
                    self.report('Relaxation is converged, finishing with a final static calculation.')
        else:
            if self.is_verbose():
                self.report('Convergence checking is not enabled - finishing with a final static calculation.')
        self.ctx.current_restart_folder = workchain.outputs.remote_folder

        self.ctx.is_converged = converged
        return self.exit_codes.NO_ERROR  # pylint: disable=no-member

    def check_shape_convergence(self, delta):
        """Check the difference in cell shape before / after the last iteratio against a tolerance."""
        threshold_angles = self.ctx.relax_settings.convergence_shape_angles
        threshold_lengths = self.ctx.relax_settings.convergence_shape_lengths

        lengths_converged = bool(delta.cell_lengths.max() <= threshold_lengths)
        if not lengths_converged:
            self.report('cell lengths changed by max {:.4g}, tolerance is {:.4g} - NOT OK'.format(delta.cell_lengths.max(),
                                                                                                  threshold_lengths))
        elif self.is_verbose():
            self.report('cell lengths changed by max {:.4g}, tolerance is {:.4g} - OK'.format(delta.cell_lengths.max(), threshold_lengths))

        angles_converged = bool(delta.cell_angles.max() <= threshold_angles)
        if not angles_converged:
            self.report('cell angles changed by max {:.4g}, tolerance is {:.4g} - NOT OK'.format(delta.cell_angles.max(), threshold_angles))
        elif self.is_verbose():
            self.report('cell angles changed by max {:.4g}, tolerance is {:.4g} - OK'.format(delta.cell_angles.max(), threshold_angles))

        return bool(lengths_converged and angles_converged)

    def check_volume_convergence(self, delta):
        """Check the convergence of the volume, given a cutoff."""
        threshold = self.ctx.relax_settings.convergence_volume
        volume_converged = bool(delta.volume <= threshold)
        if not volume_converged:
            self.report('cell volume changed by {:.4g}, tolerance is {:.4g} - NOT OK'.format(delta.volume, threshold))
        elif self.is_verbose():
            self.report('cell volume changed by {:.4g}, tolerance is {:.4g} - OK'.format(delta.volume, threshold))

        return volume_converged

    def check_positions_convergence(self, delta):
        """Check the convergence of the atomic positions, given a cutoff."""
        threshold = self.ctx.relax_settings.convergence_positions
        try:
            positions_converged = bool(np.nanmax(delta.pos_lengths) <= threshold)
        except RuntimeWarning:
            # Here we encountered the case of having one atom centered at the origin, so
            # we do not know if it is converged, so settings it to False
            # BONAN: this should never happen now - remove it later
            self.report('there is NaN entries in the relative comparison for '
                        'the positions during relaxation, assuming position is not converged')
            positions_converged = False

        if not positions_converged:
            try:
                self.report('max site position change is {:.4g}, tolerance is {:.4g} - NOT OK'.format(
                    np.nanmax(delta.pos_lengths), threshold))
            except RuntimeWarning:
                pass
        elif self.is_verbose():
            try:
                self.report('max site position change is {:.4g}, tolerance is {:.4g} - OK'.format(np.nanmax(delta.pos_lengths), threshold))
            except RuntimeWarning:
                pass

        return positions_converged

    def store_relaxed(self):
        """Store the relaxed structure."""
        workchain = self.ctx.workchains[-1]

        relaxed_structure = workchain.outputs.structure
        if self.is_verbose():
            self.report("attaching the node {}<{}> as '{}'".format(relaxed_structure.__class__.__name__, relaxed_structure.pk,
                                                                   'relax.structure'))
        self.out('relax.structure', relaxed_structure)

        if not self.ctx.is_converged:
            # If the relaxation is not converged, there is no point to
            # proceed furthure, and the result of last calculation is attached
            workchain = self.ctx.workchains[-1]
            self.out_many(self.exposed_outputs(workchain, self._base_workchain))
            return self.exit_codes.ERROR_RELAX_NOT_CONVERGED  # pylint: disable=no-member

    def results(self):
        """
        Attach the remaining output results.
        This can either be the final static calculation or the last relaxation if the
        former is not needed.

        As a final check - check if the `maximum_force` is lower than the predefined value.
        """
        workchain = self.ctx.workchains[-1]
        self.out_many(self.exposed_outputs(workchain, self._base_workchain))

        # Try to get the smearing type, there is no point to perform check if the tetrahedral smearing is used.
        if not detect_tetrahedral_method(workchain.inputs.parameters.get_dict()):
            max_force_threshold = self.ctx.relax_settings.get('force_cutoff', 0.03)
            actual_max_force = workchain.outputs.misc['maximum_force']
            if actual_max_force > max_force_threshold * 1.5:
                return self.exit_codes.ERROR_FINAL_SCF_HAS_RESIDUAL_FORCE  # pylint: disable=no-member
        else:
            self.report('Unable to presure final check for maximum force, as the tetrahedral method is used for integration.')

    def finalize(self):
        """
        Finalize the workchain.
        Clean the remote working directories of the called calcjobs
        """
        rlx_settings = self.ctx.relax_settings

        # Fence this section to avoid unnecessary process exceptions.
        try:
            if rlx_settings.get('reuse') and rlx_settings.get('clean_reuse', True):
                self.report('Cleaning remote working directory for the called CalcJobs.')
                cleaned_calcs = []
                qbd = orm.QueryBuilder()
                qbd.append(orm.WorkChainNode, filters={'id': self.node.pk})

                # Options for keeping the single point calculation work directory
                if rlx_settings.get('keep_sp_workdir', False):
                    qbd.append(orm.WorkChainNode, filters={'id': {'in': [node.pk for node in self.ctx.workchains]}})
                else:
                    qbd.append(orm.WorkChainNode,
                               edge_filters={'label': {
                                   'like': 'relax_%'
                               }},
                               filters={'id': {
                                   'in': [node.pk for node in self.ctx.workchains]
                               }})

                qbd.append(orm.CalcJobNode)

                # Find the CalcJobs to clean
                if qbd.count() > 0:
                    calcjobs = [tmp[0] for tmp in qbd.all()]
                else:
                    self.report('Cannot found called CalcJobNodes to clean.')
                    return
                # Clean the remote directories one by one
                for calculation in calcjobs:
                    try:
                        calculation.outputs.remote_folder._clean()  # pylint: disable=protected-access
                        cleaned_calcs.append(calculation.pk)
                    except BaseException:
                        pass

                if cleaned_calcs:
                    self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))  # pylint: disable=not-callable
        except BaseException as exception:
            self.report('Exception occurred during the cleaning of the remote contents: {}'.format(exception.args))

    def perform_relaxation(self):
        """Check if a relaxation is to be performed."""
        return self.ctx.relax_settings.perform

    def should_run_static_calculation(self):
        """Control whether the static calculation should be run"""
        # If the relaxation itself by passed - run the final static calculation
        if not self.perform_relaxation():
            return True

        # If the shape or volume is relaxed - we must do the final calculation
        relax_settings = self.ctx.relax_settings
        if relax_settings.shape or relax_settings.volume:
            return True
        # Otherwise, we skip the final calculation
        return False

    def is_verbose(self):
        """Are we in the verbose mode?"""
        return self.ctx.get('verbose', self._verbose)

    @classproperty
    def relax_option_class(cls):  # pylint: disable=no-self-argument
        """Class for relax options"""
        return RelaxOptions


def compare_structures(structure_a, structure_b):
    """Compare two StructreData objects A, B and return a delta (A - B) of the relevant properties."""

    delta = AttributeDict()
    delta.absolute = AttributeDict()
    delta.relative = AttributeDict()
    volume_a = structure_a.get_cell_volume()
    volume_b = structure_b.get_cell_volume()
    delta.absolute.volume = np.absolute(volume_a - volume_b)
    delta.relative.volume = np.absolute(volume_a - volume_b) / volume_a

    # Check the change in positions taking account of the pbc
    atoms_a = structure_a.get_ase()
    atoms_a.set_pbc(True)
    atoms_b = structure_b.get_ase()
    atoms_b.set_pbc(True)
    n_at = len(atoms_a)
    atoms_a.extend(atoms_b)
    pos_change_abs = np.zeros((n_at, 3))
    for isite in range(n_at):
        dist = atoms_a.get_distance(isite, isite + n_at, mic=True, vector=True)
        pos_change_abs[isite] = dist

    pos_a = np.array([site.position for site in structure_a.sites])
    # pos_b = np.array([site.position for site in structure_b.sites])
    delta.absolute.pos = pos_change_abs

    site_vectors = [delta.absolute.pos[i, :] for i in range(delta.absolute.pos.shape[0])]
    a_lengths = np.linalg.norm(pos_a, axis=1)
    delta.absolute.pos_lengths = np.array([np.linalg.norm(vector) for vector in site_vectors])
    delta.relative.pos_lengths = np.array([np.linalg.norm(vector) for vector in site_vectors]) / a_lengths

    cell_lengths_a = np.array(structure_a.cell_lengths)
    delta.absolute.cell_lengths = np.absolute(cell_lengths_a - np.array(structure_b.cell_lengths))
    delta.relative.cell_lengths = np.absolute(cell_lengths_a - np.array(structure_b.cell_lengths)) / cell_lengths_a

    cell_angles_a = np.array(structure_a.cell_angles)
    delta.absolute.cell_angles = np.absolute(cell_angles_a - np.array(structure_b.cell_angles))
    delta.relative.cell_angles = np.absolute(cell_angles_a - np.array(structure_b.cell_angles)) / cell_angles_a

    return delta


class RelaxOptions(OptionHolder):
    """
    Options for relaxations
    """
    _allowed_options = ('algo', 'energy_cutoff', 'force_cutoff', 'steps', 'positions', 'shape', 'volume', 'convergence_on',
                        'convergence_mode', 'convergence_volume', 'convergence_absolute', 'convergence_max_iterations',
                        'convergence_positions', 'convergence_shape_lengths', 'convergence_shape_angles', 'perform', 'reuse',
                        'hybrid_calc_bootstrap', 'hybrid_calc_bootstrap_wallclock', 'clean_reuse', 'keep_sp_workdir')
    _allowed_empty_fields = ('energy_cutoff', 'force_cutoff', 'hybrid_calc_bootstrap', 'hybrid_calc_bootstrap_wallclock'
                            )  # Either one of them should be set if convergence is on

    algo = typed_field('algo', (str,), 'The algorithm to use for relaxation.', 'cg')
    energy_cutoff = typed_field('energy_cutoff', (float,), """
    The cutoff that determines when the relaxation is stopped (eg. EDIFF)
    """, None)
    force_cutoff = typed_field(
        'force_cutoff', (float,), """
            The cutoff that determines when the relaxation procedure is stopped. In this
            case it stops when all forces are smaller than than the
            supplied value.
    """, 0.03)
    steps = typed_field('steps', (int,), 'Number of relaxation steps to perform (eg. NSW)', 60)
    positions = typed_field('positions', (bool,), 'If True, perform relaxation of the atomic positions', True)
    shape = typed_field('shape', (bool,), 'If True, perform relaxation of the cell shape', True)
    volume = typed_field('volume', (bool,), 'If True, perform relaxation of the cell volume', True)

    convergence_on = typed_field('convergence_on', (bool,), 'If True, perform convergence checks within the workchain', True)
    convergence_absolute = typed_field('convergence_absolute', (bool,), 'If True, use absolute value for convergence checks where possible',
                                       False)
    convergence_max_iterations = typed_field('convergence_max_iterations', (int,), 'Maximum iterations for convergence checking', 5)
    convergence_volume = typed_field('convergence_volume', (float,),
                                     'The cutoff value for the convergence check on volume, between input and output structure', 0.01)
    convergence_positions = typed_field('convergence_positions', (float,),
                                        ('The cutoff value for the convergence check on positions, in AA regardless'
                                         'of the value of ``convergence_absolute``.'
                                         'NOTE: This check is done between input and output structure.'), 0.1)
    convergence_shape_lengths = typed_field(
        'convergence_shape_lengths', (float,),
        'The cut off value for the convergence check on the lengths of the unit cell vectors, between input and output structure.', 0.1)
    convergence_shape_angles = typed_field(
        'convergence_shape_angles', (float,),
        'The cut off value for the convergence check on the angles of the unit cell vectors, between input and output structure.', 0.1)
    convergence_mode = typed_field('convergence_mode', (str,), 'Mode of the convergence - select from \'inout\' and \'last\'', 'inout')
    reuse = typed_field('reuse', (bool,), 'Whether reuse the previous calculation by copying over the remote folder.', False)
    clean_reuse = typed_field('clean_reuse', (bool,), 'Whether to perform a final cleaning of the reused calculations.', True)
    keep_sp_workdir = typed_field('keep_sp_workdir', (bool,), 'Whether to keep the workdir of the final singlepoint calculation', False)
    perform = typed_field(
        'perform',
        (bool,),
        'Wether to perform the relaxation or not',
        True,
    )
    hybrid_calc_bootstrap = typed_field('hybrid_calc_bootstrap', (bool,),
                                        'Wether to bootstrap hybrid calculation by performing standard DFT first', None)
    hybrid_calc_bootstrap_wallclock = typed_field('hybrid_calc_bootstrap_wallclock', (int,),
                                                  'Wallclock limit in second for the bootstrap calculation', None)

    @classmethod
    def validate_dict(cls, input_dict, port=None):
        """Validate the input dictionary"""
        super().validate_dict(input_dict, port)
        if isinstance(input_dict, orm.Dict):
            input_dict = input_dict.get_dict()
        force_cut = input_dict.get('force_cutoff')
        energy_cut = input_dict.get('energy_cutoff')
        if force_cut is None and energy_cut is None:
            raise InputValidationError("Either 'force_cutoff' or 'energy_cutoff' should be supplied")
        if (force_cut is not None) and (energy_cut is not None):
            raise InputValidationError("Cannot set both 'force_cutoff' and 'energy_cutoff'")


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
    return orm.Dict(dict=pydict)


def get_step_structure(traj, step):
    """Get the step structure, but assume the positions are fractional"""
    _, _, cell, symbols, positions, _ = traj.get_step_data(step)
    # Convert to cartesian coorindates
    positions = positions @ cell
    structure = orm.StructureData(cell=cell)
    for _s, _p in zip(symbols, positions):
        # Automatic species generation
        structure.append_atom(symbols=_s, position=_p)
    return structure


def detect_tetrahedral_method(input_dict: dict) -> bool:
    """
    Check if the tetrahedral method is used for BZ integration.
    """
    incar = input_dict.get('incar', {})
    if incar.get('ismear', 1) == -5:
        return True
    if input_dict.get('smearing', {}).get('tetra') and not incar.get('ismear'):
        return True
    return False
