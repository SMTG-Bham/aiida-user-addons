"""
Module to provide dryrun functionality
"""
import shutil
import numpy as np

from aiida.engine.processes.builder import ProcessBuilder
from aiida.engine import run_get_node

from aiida_vasp.calcs.vasp import VaspCalculation
from aiida_user_addons.cmdline.cmd_vasp_dryrun import vasp_dryrun as _vasp_dryrun

from .optparallel import JobScheme


def dryrun_vasp(input_dict, vasp_exe='vasp_std', timeout=10, work_dir=None, keep=False):
    """
    Perform a dryrun for a VASP calculation, return obtained information

    Args:
        input_dict (dict): The input dictionary/builder for `VaspCalculation`.
        vasp_exe (str): The VASP executable to be used.
        timeout (int): Timeout for the underlying VASP process in seconds.
        work_dir (str): Working directory, if not supply, will use a temporary directory.
        keep (bool): Wether to keep the dryrun output.

    Returns:
        dict: A dictionary of the dry run results parsed from OUTCAR
    """
    # Deal with passing an process builder
    if isinstance(input_dict, ProcessBuilder):
        try:
            output_node = prepare_inputs(input_dict)
        except Exception as error:
            raise error
    else:
        try:
            output_node = prepare_inputs(input_dict)
        except Exception as error:
            raise error

    folder = output_node.dry_run_info['folder']
    outcome = _vasp_dryrun(folder, vasp_exe=vasp_exe, timeout=timeout, work_dir=work_dir, keep=keep)
    if not keep:
        shutil.rmtree(folder)

    return outcome


def get_jobscheme(input_dict, nprocs, vasp_exe='vasp_std', **kwargs):
    """
    Perform a dryrun for the input and workout the best parallelisation strategy
    Args:
        input_dict (dict,ProcessBuilder): Inputs of the VaspCalculation
        nprocs (int): Target number of processes to be used
        vasp_exe (str): The executable of local VASP program to be used
        kwargs: Addition keyword arguments to be passed to `JobScheme`

    Returns:
        int, int: The KPAR and NCORE that should be used

    """
    dryout = dryrun_vasp(input_dict, vasp_exe)
    scheme = JobScheme.from_dryrun(dryout, nprocs, **kwargs)
    return scheme


def prepare_inputs(inputs):
    """Prepare inputs"""

    # Have to turn store_provenance to False
    inputs = dict(inputs)
    inputs['metadata'] = dict(inputs['metadata'])
    inputs['metadata']['store_provenance'] = False
    inputs['metadata']['dry_run'] = True
    vasp = VaspCalculation(inputs=inputs)
    from aiida.common.folders import SubmitTestFolder
    from aiida.engine.daemon.execmanager import upload_calculation
    from aiida.transports.plugins.local import LocalTransport

    with LocalTransport() as transport:
        with SubmitTestFolder() as folder:
            calc_info = vasp.presubmit(folder)
            transport.chdir(folder.abspath)
            upload_calculation(vasp.node, transport, calc_info, folder, inputs=vasp.inputs, dry_run=True)
            vasp.node.dry_run_info = {'folder': folder.abspath, 'script_filename': vasp.node.get_option('submit_script_filename')}
    return vasp.node


def dryrun_relax_builder(builder, **kwargs):
    """Dry run a relaxation workchain builder"""
    from aiida_vasp.data.potcar import PotcarData
    from aiida.orm import KpointsData, Dict
    vasp_builder = VaspCalculation.get_builder()

    # Setup the builder for the bare calculation
    vasp_builder.code = builder.vasp.code
    vasp_builder.parameters = Dict(dict=builder.vasp.parameters.get_dict()['vasp'])
    if builder.vasp.kpoints is not None:
        vasp_builder.kpoints = builder.vasp.kpoints
    else:
        vasp_builder.kpoints = KpointsData()
        vasp_builder.kpoints.set_cell_from_structure(builder.structure)
        vasp_builder.kpoints.set_kpoints_mesh_from_density(builder.vasp.kpoints_spacing.value * np.pi * 2)
    vasp_builder.metadata.options = builder.vasp.options.get_dict()  # pylint: disable=no-member
    vasp_builder.potential = PotcarData.get_potcars_from_structure(builder.structure, builder.vasp.potential_family.value,
                                                                   builder.vasp.potential_mapping.get_dict())
    vasp_builder.structure = builder.structure

    return dryrun_vasp(vasp_builder, **kwargs)


def dryrun_vaspu_builder(builder, **kwargs):
    """Dry run a vaspu.vasp workchain builder"""
    from aiida_vasp.data.potcar import PotcarData
    from aiida.orm import KpointsData, Dict
    vasp_builder = VaspCalculation.get_builder()

    # Setup the builder for the bare calculation
    vasp_builder.code = builder.code
    vasp_builder.parameters = Dict(dict=builder.parameters.get_dict()['vasp'])
    if builder.kpoints is not None:
        vasp_builder.kpoints = builder.kpoints
    else:
        vasp_builder.kpoints = KpointsData()
        vasp_builder.kpoints.set_cell_from_structure(builder.structure)
        vasp_builder.kpoints.set_kpoints_mesh_from_density(builder.kpoints_spacing.value * np.pi * 2)
    vasp_builder.metadata.options = builder.options.get_dict()  # pylint: disable=no-member
    vasp_builder.potential = PotcarData.get_potcars_from_structure(builder.structure, builder.potential_family.value,
                                                                   builder.potential_mapping.get_dict())
    vasp_builder.structure = builder.structure

    return dryrun_vasp(vasp_builder, **kwargs)
