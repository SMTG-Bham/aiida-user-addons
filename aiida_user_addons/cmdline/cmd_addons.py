"""
Additional CLI scripts
"""
import click
from tqdm import tqdm

from aiida_user_addons.tools.scfcheck import database_sweep
from aiida.cmdline.commands.cmd_data import verdi_data
from aiida.cmdline.params.arguments import PROCESS, WORKFLOW, CALCULATION
from aiida.cmdline.utils.echo import echo_success, echo_info, echo_error


@verdi_data.group('addons')
def addons():
    """Entry point for commands under aiida-user-addons"""


@addons.command('check-nelm')
@click.option('--reset', is_flag=True, help='Remove all `nelm_breach` tags')
def check_nelm(reset):
    """
    Perform a sweep to check if any VaspCalculation have unconverged electronic
    structure but has exit_status = 0.
    The output of these calculations are not reliable. This usually does not cause
    errors as higher level of workchain will check for forces etc.

    A `nelm_breach` will be added to the `extras` for the calculations examined.
    Only those that do not have `nelm_breach` tag will be checked
    """
    from aiida.orm import QueryBuilder
    from aiida.plugins import CalculationFactory
    Vasp = CalculationFactory('vasp.vasp')
    if reset:
        query = QueryBuilder()
        query.append(Vasp, filters={'extras': {'has_key': 'nelm_breach'}})
        total = query.count()
        for (node,) in tqdm(query.iterall(), total=total):
            node.delete_extra('nelm_breach')
    else:
        database_sweep()


@addons.command('export_vasp')
@PROCESS('process')
@click.argument('folder')
@click.option('--include-potcar', default=False, is_flag=True, help='Wether to include POTCAR in the export folder')
@click.option('--decompress', default=False, is_flag=True, help='Wether to decompress the contents')
def export_vasp(process, folder, decompress, include_potcar):
    """Export a VASP calculation, works for both `VaspCalculation` or `VaspWorkChain`"""
    from aiida_user_addons.tools.vasp import export_vasp_calc
    export_vasp_calc(process, folder, decompress=decompress, include_potcar=include_potcar)


@addons.command('export_relax')
@WORKFLOW('workflow')
@click.argument('folder')
@click.option('--include-potcar', default=False, is_flag=True, help='Wether to include POTCAR in the export folder')
@click.option('--decompress', default=False, is_flag=True, help='Wether to decompress the contents')
def export_relax(workflow, folder, decompress, include_potcar):
    """Export a VASP relaxation workflow"""
    from aiida_user_addons.tools.vasp import export_relax as _export_relax
    _export_relax(workflow, folder, decompress=decompress, include_potcar=include_potcar)


@addons.command('remotecat')
@CALCULATION('calcjob')
@click.argument('fname')
@click.option('--save-to', '-s', help='Name of the file to save to')
def remotecat(calcjob, fname, save_to):
    """Cat the content of an file lying on the remote folder of the calculation"""
    import tempfile
    from shutil import copyfileobj
    import os
    import sys

    rfolder = calcjob.outputs.remote_folder
    if save_to is None:
        fd, temppath = tempfile.mkstemp()
    else:
        temppath = save_to

    rfolder.getfile(fname, temppath)

    with open(temppath, 'rb') as fhandle:
        copyfileobj(fhandle, sys.stdout.buffer)

    if save_to is None:
        os.close(fd)
        os.remove(temppath)


@addons.command('remotepull')
@CALCULATION('calcjob')
@click.argument('dest')
@click.option('--max-size', '-m', help='Maximum size of the files to be retrieved - this is passed to rsync')
def remotepull(calcjob, dest, max_size):
    """Cat the content of an file lying on the remote folder of the calculation"""
    import subprocess
    rfolder = calcjob.outputs.remote_folder
    cmd_args = ['rsync', '-av']

    if max_size:
        cmd_args.extend(['--max-size', max_size])

    cmd_args.append('{}:{}/'.format(rfolder.computer.hostname, rfolder.get_remote_path()))
    if not dest.endswith('/'):
        dest = dest + '/'
    cmd_args.append(dest)

    echo_info('Running commands: {}'.format(' '.join(cmd_args)))

    completed = subprocess.run(cmd_args)
    if completed.returncode != 0:
        echo_error('Failled to pull data using rsync')
    else:
        echo_success('Remote folder pulled to {}'.format(dest))
