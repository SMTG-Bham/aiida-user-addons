"""
VASP workchain.

---------------
Contains the VaspWorkChain class definition which uses the BaseRestartWorkChain.
"""
import numpy as np

from aiida.engine import while_, if_
from aiida.common.lang import override
#from aiida.engine.job_processes import override
from aiida.common.extendeddicts import AttributeDict
from aiida.common.exceptions import NotExistent, InputValidationError
from aiida.common.utils import classproperty
from aiida.plugins import CalculationFactory
from aiida.orm import Code, KpointsData, Dict
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.engine.processes.workchains.restart import BaseRestartWorkChain

from aiida_vasp.workchains.vasp import VaspWorkChain as VanillaVaspWorkChain

from aiida_vasp.utils.aiida_utils import get_data_class, get_data_node
from aiida_vasp.utils.workchains import compose_exit_code
from aiida_vasp.utils.workchains import prepare_process_inputs
from aiida_vasp.assistant.parameters import ParametersMassage

from aiida_user_addons.common.inputset.vaspsets import VASPInputSet
from ..common.inputset.vaspsets import get_ldau_keys

from .common import parameters_validator

assert issubclass(VanillaVaspWorkChain,
                  BaseRestartWorkChain), 'vasp.vasp workchain is not a subclass of BaseRestartWorkChain from aiida-core'


class VaspWorkChain(VanillaVaspWorkChain):
    """
    The VASP workchain.

    -------------------
    Error handling enriched wrapper around VaspCalculation.

    Deliberately conserves most of the interface (required inputs) of the VaspCalculation class, but
    makes it possible for a user to interact with a workchain and not a calculation.

    This is intended to be used instead of directly submitting a VaspCalculation,
    so that future features like
    automatic restarting, error checking etc. can be propagated to higher level workchains
    automatically by implementing them here.

    Usage::

        from aiida.common.extendeddicts import AttributeDict
        from aiida.work import submit
        basevasp = WorkflowFactory('vasp.vasp')
        inputs = basevasp.get_builder()
        inputs = AttributeDict()
        ## ... set inputs
        submit(basevasp, **inputs)

    To see a working example, including generation of input nodes from scratch, please
    refer to ``examples/run_vasp_lean.py``.


    Additional functionalities:

    - Automatic setting LDA+U key using the ``ldau_mapping`` input port.

    - Set kpoints using spacing in A^-1 * 2pi with the ``kpoints_spacing`` input port.

    - Perform dryrun and set parameters such as KPAR and NCORE automatically if ``auto_parallel`` input port exists.
      this will give rise to an additional output node ``parallel_settings`` containing the strategy obtained.

    """
    _verbose = False
    _calculation = CalculationFactory('vasp.vasp')

    @classmethod
    def define(cls, spec):
        super(VaspWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=(get_data_class('structure'), get_data_class('cif')), required=True)
        spec.input('kpoints', valid_type=get_data_class('array.kpoints'), required=False)
        spec.input('potential_family', valid_type=get_data_class('str'), required=True, serializer=to_aiida_type)
        spec.input('potential_mapping', valid_type=get_data_class('dict'), required=True, serializer=to_aiida_type)
        spec.input('parameters', valid_type=get_data_class('dict'), required=True, validator=parameters_validator)
        spec.input('options', valid_type=get_data_class('dict'), required=True, serializer=to_aiida_type)
        spec.input('settings', valid_type=get_data_class('dict'), required=False, serializer=to_aiida_type)
        spec.input('wavecar', valid_type=get_data_class('vasp.wavefun'), required=False)
        spec.input('chgcar', valid_type=get_data_class('vasp.chargedensity'), required=False)
        spec.input('restart_folder',
                   valid_type=get_data_class('remote'),
                   required=False,
                   help="""
            The restart folder from a previous workchain run that is going to be used.
            """)
        spec.input('max_iterations',
                   valid_type=get_data_class('int'),
                   required=False,
                   default=lambda: get_data_node('int', 5),
                   serializer=to_aiida_type,
                   help="""
            The maximum number of iterations to perform.
            """)
        spec.input('clean_workdir',
                   valid_type=get_data_class('bool'),
                   required=False,
                   serializer=to_aiida_type,
                   default=lambda: get_data_node('bool', True),
                   help="""
            If True, clean the work dir upon the completion of a successfull calculation.
            """)
        spec.input('verbose',
                   valid_type=get_data_class('bool'),
                   required=False,
                   serializer=to_aiida_type,
                   default=lambda: get_data_node('bool', False),
                   help="""
            If True, enable more detailed output during workchain execution.
            """)
        spec.input('ldau_mapping',
                   valid_type=get_data_class('dict'),
                   required=False,
                   serializer=to_aiida_type,
                   help="Mappings, see the doc string of 'get_ldau_keys'")
        spec.input('kpoints_spacing',
                   valid_type=get_data_class('float'),
                   required=False,
                   serializer=to_aiida_type,
                   help='Spacing for the kpoints in units A^-1 * 2pi')
        spec.input('auto_parallel',
                   valid_type=get_data_class('dict'),
                   serializer=to_aiida_type,
                   required=False,
                   help='Automatic parallelisation settings, keywords passed to `get_jobscheme` function.')
        spec.input('dynamics.positions_dof',
                   valid_type=get_data_class('list'),
                   serializer=to_aiida_type,
                   required=False,
                   help="""
            Site dependent flag for selective dynamics when performing relaxation
            """)
        spec.outline(
            cls.setup,
            cls.init_inputs,
            if_(cls.run_auto_parallel)(
                cls.prepare_inputs,
                cls.perform_autoparallel
            ),
            while_(cls.should_run_process)(
                cls.prepare_inputs,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )  # yapf: disable
        spec.output('parallel_settings', valid_type=get_data_class('dict'), required=False)

    def init_inputs(self):
        """Make sure all the required inputs are there and valid, create input dictionary for calculation."""

        output = super().init_inputs()
        if output is not None and output.status != 0:
            return output

        # Set the kpoints (kpoints)
        if 'kpoints' in self.inputs:
            self.ctx.inputs.kpoints = self.inputs.kpoints
        elif 'kpoints_spacing' in self.inputs:
            kpoints = KpointsData()
            kpoints.set_cell_from_structure(self.ctx.inputs.structure)
            kpoints.set_kpoints_mesh_from_density(self.inputs.kpoints_spacing.value * np.pi * 2)
            self.ctx.inputs.kpoints = kpoints
        else:
            raise InputValidationError("Must supply either 'kpoints' or 'kpoints_spacing'")

        # Setup LDAU keys
        if 'ldau_mapping' in self.inputs:
            ldau_settings = self.inputs.ldau_mapping.get_dict()
            ldau_keys = get_ldau_keys(self.ctx.inputs.structure, **ldau_settings)
            # Directly update the raw inputs passed to VaspCalculation
            self.ctx.inputs.parameters.update(ldau_keys)

    def run_auto_parallel(self):
        """Wether we should run auto-parallelisation test"""
        return 'auto_parallel' in self.inputs and self.inputs.auto_parallel.value is True

    def perform_autoparallel(self):
        """Dry run and obtain the best parallelisation settings"""
        from aiida_user_addons.tools.dryrun import get_jobscheme
        self.report(f'Performing local dryrun for auto-parallelisation')  # pylint: disable=not-callable

        ind = prepare_process_inputs(self.ctx.inputs)

        nprocs = self.ctx.inputs.metadata['options']['resources']['tot_num_mpiprocs']

        # Take the settings pass it to the function
        kwargs = self.inputs.auto_parallel.get_dict()
        if 'cpus_per_node' not in kwargs:
            kwargs['cpus_per_node'] = self.inputs.code.computer.get_default_mpiprocs_per_machine()

        # If the dryrun errored, proceed the workchain
        try:
            scheme = get_jobscheme(ind, nprocs, **kwargs)
        except Exception as error:
            self.report(f'Dry-run errorred, process with cautions, message: {error.args}')  # pylint: disable=not-callable
            return

        if (scheme.ncore is None) or (scheme.kpar is None):
            self.report(f'Error NCORE: {scheme.ncore}, KPAR: {scheme.kpar}')  # pylint: disable=not-callable
            return

        parallel_opts = {'ncore': scheme.ncore, 'kpar': scheme.kpar}
        self.report(f'Found optimum KPAR={scheme.kpar}, NCORE={scheme.ncore}')  # pylint: disable=not-callable
        self.ctx.inputs.parameters.update(parallel_opts)
        self.out('parallel_settings', Dict(dict={'ncore': scheme.ncore, 'kpar': scheme.kpar}).store())
