"""
Additional CLI scripts
"""
import click
from tqdm import tqdm

from aiida_user_addons.tools.scfcheck import database_sweep
from aiida.cmdline.commands.cmd_data import verdi_data
from aiida.cmdline.params.arguments import PROCESS, WORKFLOW


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
    from aiida_user_addons.tools.vasp import export_relax
    export_relax(workflow, folder, decompress=decompress, include_potcar=include_potcar)
