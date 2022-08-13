"""
Pipline for in place modification of builders

```python
builder = VaspRelaxWorkChain.get_builder()
pipeline = BuilderUpdater(builder)
pipeline.use_input_set(structure, ..., ...)
pipeline.use_code(...., ...)
piplined.set_wallclock_seconds(...).set_num_machines(...)
```
"""
from pprint import pprint
from warnings import warn
from aiida_user_addons.common.inputset.vaspsets import VASPInputSet
from aiida_user_addons.vworkflows.relax import RelaxOptions
from aiida.engine.processes.builder import ProcessBuilder
import aiida.orm as orm

from .dictwrap import DictWrapper


class BuilderUpdater:
    """Base class for builder updater"""

    def __init__(self, builder: ProcessBuilder):
        """Instantiate a pipline"""
        self.builder = builder

    def show_builder(self):
        """Print stuff defined in the builder"""
        pprint(builder_to_dict(self.builder, unpack=True))


class VaspBuilderUpdater(BuilderUpdater):
    """
    Class handling updating the ProcessBuilder for `VaspWorkChain`.
    """

    def __init__(self, builder, root_namespace=None):
        super().__init__(builder)
        if root_namespace is None:
            self.root_namespace = builder
        else:
            self.root_namespace = root_namespace

        self.namespace_vasp = builder
        self.parameters_wrapped = None
        self.options_wrapped = None
        self.settings_wrapped = None

    def use_inputset(self, structure, set_name='UCLRelaxSet', overrides=None):

        inset = VASPInputSet(set_name, structure, overrides=overrides)
        self.namespace_vasp.parameters = orm.Dict(dict={'incar': inset.get_input_dict()})
        self.namespace_vasp.potential_family = orm.Str('PBE.54')
        self.namespace_vasp.potential_mapping = orm.Dict(dict=inset.get_pp_mapping())
        self.namespace_vasp.kpoints_spacing = orm.Float(0.05)
        self.root_namespace.structure = structure

        return self

    def _initialise_parameters_wrapper(self, force=False):
        """Initialise DictWrapper for tracking INCAR tags"""
        if self.parameters_wrapped is None or force:
            self.parameters_wrapped = DictWrapper(self.namespace_vasp.parameters, self.namespace_vasp, 'parameters')

    def _initialise_options_wrapper(self, force=False):
        """Initialise DictWrapper for tracking INCAR tags"""
        if self.options_wrapped is None or force:
            self.options_wrapped = DictWrapper(self.namespace_vasp.options, self.namespace_vasp, 'options')

    def _initialise_settings_wrapper(self, force=False):
        """Initialise DictWrapper for tracking INCAR tags"""
        if self.settings_wrapped is None or force:
            self.settings_wrapped = DictWrapper(self.namespace_vasp.settings, self.namespace_vasp, 'settings')

    def set_kspacing(self, kspacing: float):
        self.namespace_vasp.kpoints_spacing = orm.Float(kspacing)
        if self.namespace_vasp.kpoints:
            del self.namespace_vasp.kpoints
        return self

    @property
    def incar(self):
        """Return the INCAR dictionary"""
        return dict(self.namespace_vasp.parameters['incar'])

    @property
    def settings(self):
        """Return the wrapped settings dictionary"""
        self._initialise_settings_wrapper()
        return self.settings_wrapped

    @property
    def parameters(self):
        """Return the wrapped parameters dictionary"""
        self._initialise_parameters_wrapper()
        return self.parameters_wrapped

    @property
    def options(self):
        """Return the wrapped options dictionary"""
        self._initialise_options_wrapper()
        return self.options_wrapped

    def clear_incar(self):
        """Clear existing settings"""
        if self.namespace_vasp.parameters:
            del self.namespace_vasp.parameters
        self.parameters_wrapped = None

    def update_incar(self, *args, **kwargs):
        """Update incar tags"""
        self._initialise_parameters_wrapper()
        # Make a copy of the incar for modification
        incar = dict(self.parameters_wrapped['incar'])
        incar.update(*args, **kwargs)
        self.parameters_wrapped['incar'] = incar
        return self

    def set_code(self, code):
        self.namespace_vasp.code = code
        return self

    def set_default_options(self, **override):
        options = None

        # Try to use a sensible default from code's computer/scheduler type
        if self.namespace_vasp.code:
            computer = self.namespace_vasp.code.computer
            # Try to match with computer name
            for key in OPTIONS_TEMPLATES:
                if key in computer.label.upper():
                    options = orm.Dict(dict=OPTIONS_TEMPLATES[key])
                    break
            # Try to match with scheduler type
            if options is None:
                for key in OPTIONS_TEMPLATES:
                    if key in computer.scheduler_type.upper():
                        options = orm.Dict(dict=OPTIONS_TEMPLATES[key])
                        break

        # Use the very default settings
        if options is None:
            warn('Using default options template - adjustment needed for the target computer')
            options = orm.Dict(dict={
                'resources': {
                    'tot_num_mpiprocs': 1,
                },
                'max_wallclock_seconds': 3600,
                'import_sys_environment': False,
                **override
            })
        else:
            # Update with the overrides
            options.update_dict(override)

        self.namespace_vasp.options = options
        self._initialise_options_wrapper()
        return self

    def set_kpoints_mesh(self, mesh, offset):
        """Use mesh for kpoints"""
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.root_namespace.structure)
        kpoints.set_kpoints_mesh(mesh, offset)
        self.namespace_vasp.kpoints = kpoints
        if self.namespace_vasp.kpoints_spacing:
            del self.namespace_vasp.kpoints_spacing

    def clear_incar(self):
        """Clear existing settings"""
        if self.namespace_vasp.parameters:
            del self.namespace_vasp.parameters
        self.parameters_wrapped = None

    def update_incar(self, *args, **kwargs):
        """Update incar tags"""
        if self.namespace_vasp.parameters is None:
            self.namespace_vasp.parameters = orm.Dict(dict={'incar': {}})

        self._initialise_parameters_wrapper()
        # Make a copy of the incar for modification
        incar = dict(self.parameters_wrapped['incar'])
        incar.update(*args, **kwargs)
        self.parameters_wrapped['incar'] = incar
        return self

    def set_code(self, code):
        self.namespace_vasp.code = code
        return self

    def set_default_options(self, **override):
        options = None

        # Try to use a sensible default from code's computer/scheduler type
        if self.namespace_vasp.code:
            computer = self.namespace_vasp.code.computer
            for key in OPTIONS_TEMPLATES:
                if key in computer.label.upper():
                    new = dict(OPTIONS_TEMPLATES[key])
                    new.update(override)
                    options = orm.Dict(dict=new)
                    break
            if options is None:
                for key in OPTIONS_TEMPLATES:
                    if key in computer.scheduler_type.upper():
                        new = dict(OPTIONS_TEMPLATES[key])
                        new.update(override)
                        options = orm.Dict(dict=new)
                        break

        # Use the very default settings
        if options is None:
            warn('Using default options template - adjustment needed for the target computer')
            options = orm.Dict(dict={
                'resources': {
                    'tot_num_mpiprocs': 1,
                },
                'max_wallclock_seconds': 3600,
                'import_sys_environment': False,
                **override
            })

        self.namespace_vasp.options = options
        self._initialise_options_wrapper()
        return self

    def set_kpoints_mesh(self, mesh, offset):
        """Use mesh for kpoints"""
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.root_namespace.structure)
        kpoints.set_kpoints_mesh(mesh, offset)
        self.namespace_vasp.kpoints = kpoints
        try:
            del self.namespace_vasp.kpoints_spacing
        except KeyError:
            pass
        return self

    def update_options(self, *args, **kwargs):
        """Update the options"""
        if self.options_wrapped is None:
            self.set_default_options()
        self.options_wrapped.update(*args, **kwargs)
        return self

    def clear_options(self):
        if self.namespace_vasp.options:
            del self.namespace_vasp.options
        self.options_wrapped = None

    def update_settings(self, *args, **kwargs):
        """Update the options"""
        if self.namespace_vasp.settings is None:
            self.namespace_vasp.settings = orm.Dict(dict={})
        self._initialise_settings_wrapper()
        self.settings_wrapped.update(*args, **kwargs)
        return self

    def clear_settings(self):
        """Clear existing settings"""
        if self.namespace_vasp.settings:
            del self.namespace_vasp.settings
        self.settings_wrapped = None

    def set_label(self, label=None):
        """Set the toplevel label, default to the label of the structure"""
        if label is None:
            label = self.root_namespace.structure.label
        self.root_namespace.metadata.label = label
        return self

    def update_resources(self, *args, **kwargs):
        """Update resources"""
        if self.options_wrapped is None:
            self.set_default_options()
        resources = dict(self.options_wrapped['resources'])
        resources.update(*args, **kwargs)
        self.options_wrapped['resources'] = resources
        return self


class VaspRelaxUpdater(VaspBuilderUpdater):
    """
    Class handling updating the ProcessBuilder for `VaspRelaxWorkChain`.
    """

    def __init__(self, builder, override_vasp_namespace=None, namespace_relax=None):
        super().__init__(builder)
        # The primary VASP namespace is under builder.vasp
        if override_vasp_namespace is None:
            self.namespace_vasp = builder.vasp
        else:
            self.namespace_vasp = override_vasp_namespace

        if namespace_relax is None:
            self.namespace_relax = builder
        else:
            self.namespace_relax = namespace_relax

    def update_relax_settings(self, **kwargs):
        """Set/update RelaxOptions controlling the operation of the workchain"""
        if self.namespace_relax.relax_settings is None:
            current_options = RelaxOptions()
        else:
            current_options = RelaxOptions(**self.namespace_relax.relax_settings.get_dict())
        for key, value in kwargs.items():
            setattr(current_options, key, value)
        self.namespace_relax.relax_settings = current_options.to_aiida_dict()

    def clear_relax_settings(self):
        """Reset any existing relax options"""
        self.root_namespace.relax_settings = RelaxOptions().to_aiida_dict()


# Template for options. This is machine specific....
# TODO make this configurable via files.

OPTIONS_TEMPLATES = {
    'SGE': {
        'resources': {
            'tot_num_mpiprocs': 1,
            'parallel_env': 'mpi'
        },
        'max_wallclock_seconds': 3600,
        'import_sys_environment': False,
    },
    'FW': {
        'resources': {
            'tot_num_mpiprocs': 1,
        },
        'max_wallclock_seconds': 3600,
    },
    'SLURM': {
        'resources': {
            'num_machines': 1,
        },
        'max_wallclock_seconds': 3600,
        'import_sys_environment': False,
    },
    'ARCHER2': {
        'resources': {
            'tot_num_mpiprocs': 128,
            'num_machines': 1,
        },
        'max_wallclock_seconds': 3600,
        'import_sys_environment': False,
        'mpirun_extra_params': ['--distribution=block:block', '--hint=nomultithread'],
        'account': 'e05-power-dos',
        'queue_name': 'standard',
        'qos': 'standard',
    }
}


def builder_to_dict(builder: ProcessBuilder, unpack=True):
    """
    Convert a builder to a dictionary and unpack certain nodes.

    When unpacked, the resulting dictionary cannot be used for `submit`/`run`.

    The primary useage of the resulting dictionary is for pretty printing.

    NOTE: this is not necessary anymore with AiiDA v2?
    """
    data = {}
    for key, value in builder._data.items():
        if hasattr(value, '_data'):
            value = builder_to_dict(builder[key])
        if unpack:
            if isinstance(value, orm.Dict):
                value = value.get_dict()
            if isinstance(value, orm.List):
                value = value.get_list()
        data[key] = value
    return data


class VaspHybridBandUpdater(VaspBuilderUpdater):
    """Updater for VaspHybridBandsWorkChain"""

    def __init__(self, builder, namespace_vasp=None):
        super().__init__(builder)
        # The primary VASP namespace is under builder.vasp
        if namespace_vasp is None:
            self.namespace_vasp = builder.scf
        else:
            self.namespace_vasp = namespace_vasp


class VaspBandUpdater(VaspBuilderUpdater):
    """Updater for VaspBandsWorkChain"""

    def __init__(self, builder, namespace_vasp=None):
        super().__init__(builder)
        # The primary VASP namespace is under builder.vasp
        if namespace_vasp is None:
            self.namespace_vasp = builder.scf
        else:
            self.namespace_vasp = namespace_vasp
