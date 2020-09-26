"""
Module to provide dryrun functionality
"""
import shutil

from aiida.engine.processes.builder import ProcessBuilder
from aiida.engine import run_get_node

from aiida_vasp.calcs.vasp import VaspCalculation
from aiida_user_addons.cmdline.cmd_vasp_dryrun import vasp_dryrun as _vasp_dryrun


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
        input_dict['metadata']['store_provenance'] = False
        input_dict['metadata']['dry_run'] = True
        try:
            output_node = run_get_node(input_dict).node
        except Exception as error:
            raise error
        finally:
            input_dict['metadata']['store_provenance'] = True
            input_dict['metadata']['dry_run'] = False
    else:
        input_dict['metadata']['store_provenance'] = False
        input_dict['metadata']['dry_run'] = True
        try:
            output_node = run_get_node(VaspCalculation, **input_dict).node
        except Exception as error:
            raise error
        finally:
            input_dict['metadata']['store_provenance'] = True
            input_dict['metadata']['dry_run'] = False

    info = output_node.dry_run_info
    folder = info['folder']
    outcome = _vasp_dryrun(folder, vasp_exe=vasp_exe, timeout=timeout, work_dir=work_dir, keep=keep)
    shutil.rmtree(folder)

    return outcome
