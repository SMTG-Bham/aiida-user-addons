"""
Additional CLI scripts
"""
import click
from aiida.cmdline.commands.cmd_data import verdi_data
from aiida.cmdline.params.arguments import (
    CALCULATION,
    PROCESS,
    WORKFLOW,
)
from aiida.cmdline.utils.echo import (
    echo_critical,
    echo_error,
    echo_info,
    echo_success,
)
from tqdm import tqdm

from aiida_user_addons.tools.scfcheck import database_sweep


@verdi_data.group("addons")
def addons():
    """Entry point for commands under aiida-user-addons"""


@addons.command("check-nelm")
@click.option("--reset", is_flag=True, help="Remove all `nelm_breach` tags")
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

    Vasp = CalculationFactory("vasp.vasp")
    if reset:
        query = QueryBuilder()
        query.append(Vasp, filters={"extras": {"has_key": "nelm_breach"}})
        total = query.count()
        for (node,) in tqdm(query.iterall(), total=total):
            node.delete_extra("nelm_breach")
    else:
        database_sweep()


@addons.command("export_vasp")
@PROCESS("process")
@click.argument("folder")
@click.option(
    "--include-potcar",
    default=False,
    is_flag=True,
    help="Wether to include POTCAR in the export folder",
)
@click.option(
    "--decompress",
    default=False,
    is_flag=True,
    help="Wether to decompress the contents",
)
def export_vasp(process, folder, decompress, include_potcar):
    """Export a VASP calculation, works for both `VaspCalculation` or `VaspWorkChain`"""
    from aiida_user_addons.tools.vasp import export_vasp_calc

    export_vasp_calc(
        process, folder, decompress=decompress, include_potcar=include_potcar
    )


@addons.command("export_relax")
@WORKFLOW("workflow")
@click.argument("folder")
@click.option(
    "--include-potcar",
    default=False,
    is_flag=True,
    help="Wether to include POTCAR in the export folder",
)
@click.option(
    "--decompress",
    default=False,
    is_flag=True,
    help="Wether to decompress the contents",
)
def export_relax(workflow, folder, decompress, include_potcar):
    """Export a VASP relaxation workflow"""
    from aiida_user_addons.tools.vasp import (
        export_relax as _export_relax,
    )

    _export_relax(
        workflow, folder, decompress=decompress, include_potcar=include_potcar
    )


@addons.command("remotecat")
@CALCULATION("calcjob")
@click.argument("fname")
@click.option("--save-to", "-s", help="Name of the file to save to")
def remotecat(calcjob, fname, save_to):
    """
    Print the conetent of a remote file to STDOUT

    This command for printing the content of a remote file to STDOUT.
    Useful for analysing running calculations.
    """

    import os
    import sys
    import tempfile
    from shutil import copyfileobj

    rfolder = calcjob.outputs.remote_folder
    if save_to is None:
        fd, temppath = tempfile.mkstemp()
    else:
        temppath = save_to

    rfolder.getfile(fname, temppath)

    with open(temppath, "rb") as fhandle:
        copyfileobj(fhandle, sys.stdout.buffer)

    if save_to is None:
        os.close(fd)
        os.remove(temppath)


@addons.command("remotepull")
@CALCULATION("calcjob")
@click.argument("dest")
@click.option(
    "--max-size",
    "-m",
    help="Maximum size of the files to be retrieved - this is passed to rsync",
)
def remotepull(calcjob, dest, max_size):
    """
    Pull a calculation folder from the remote

    This command for pull a calculation folder to a local folder.
    `rsync` is used for doing the heavy lifting.
    """
    import subprocess

    rfolder = calcjob.outputs.remote_folder
    cmd_args = ["rsync", "-av"]

    if max_size:
        cmd_args.extend(["--max-size", max_size])

    cmd_args.append(f"{rfolder.computer.hostname}:{rfolder.get_remote_path()}/")
    if not dest.endswith("/"):
        dest = dest + "/"
    cmd_args.append(dest)

    echo_info("Running commands: {}".format(" ".join(cmd_args)))

    completed = subprocess.run(cmd_args)
    if completed.returncode != 0:
        echo_error("Failled to pull data using rsync")
    else:
        echo_success(f"Remote folder pulled to {dest}")


@addons.command("remotetail")
@CALCULATION("calcjob")
@click.argument("fname")
def remotetail(calcjob, fname):
    """
    Follow a file on the remote computer

    This command will launch a ssh session dedicated for following a file
    using the `tail -f` command
    """
    import os

    from aiida.common.exceptions import NotExistent

    try:
        transport = calcjob.get_transport()
    except NotExistent as exception:
        echo_critical(repr(exception))

    remote_workdir = calcjob.get_remote_workdir()

    if not remote_workdir:
        echo_critical(
            "no remote work directory for this calcjob, maybe the daemon did not submit it yet"
        )

    command = tailf_command(transport, remote_workdir, fname)
    os.system(command)


@addons.command("relaxcat")
@WORKFLOW("workflow")
@click.argument("fname")
def relaxcat(workflow, fname):
    """Cat the output of the last calculation of a finished workflow"""
    from aiida.cmdline.commands.cmd_calcjob import calcjob_outputcat
    from aiida.orm import CalcJobNode, QueryBuilder, WorkChainNode

    q = QueryBuilder()
    q.append(WorkChainNode, filters={"id": workflow.id})
    q.append(WorkChainNode)
    q.append(CalcJobNode, tag="calc", project=["*", "ctime"])
    q.order_by({"calc": {"ctime": "desc"}})
    calc, ctime = q.first()

    click.Context(calcjob_outputcat).invoke(calcjob_outputcat, calcjob=calc, path=fname)


def tailf_command(transport, remotedir, fname):
    """
    Specific gotocomputer string to connect to a given remote computer via
    ssh and directly go to the calculation folder and then do tail -f of the target file.
    """
    from aiida.common.escaping import escape_for_bash

    further_params = []
    if "username" in transport._connect_args:
        further_params.append(
            "-l {}".format(escape_for_bash(transport._connect_args["username"]))
        )

    if "port" in transport._connect_args and transport._connect_args["port"]:
        further_params.append("-p {}".format(transport._connect_args["port"]))

    if (
        "key_filename" in transport._connect_args
        and transport._connect_args["key_filename"]
    ):
        further_params.append(
            "-i {}".format(escape_for_bash(transport._connect_args["key_filename"]))
        )

    further_params_str = " ".join(further_params)

    connect_string = (
        """ "if [ -d {escaped_remotedir} ] ;"""
        """ then cd {escaped_remotedir} ; {bash_command} -c 'tail -f {fname}' ; else echo '  ** The directory' ; """
        """echo '  ** {remotedir}' ; echo '  ** seems to have been deleted, I logout...' ; fi" """.format(
            bash_command=transport._bash_command_str,
            escaped_remotedir=f"'{remotedir}'",
            remotedir=remotedir,
            fname=fname,
        )
    )

    cmd = "ssh -t {machine} {further_params} {connect_string}".format(
        further_params=further_params_str,
        machine=transport._machine,
        connect_string=connect_string,
    )
    return cmd
