"""
Module for dry-running a VASP calculation
"""
import click
import subprocess as sb
import tempfile
import shutil
from pathlib import Path
import time
import yaml


@click.command('vasp-dryrun')
@click.option('--input-dir',
              help='Where the VASP input is, default to the current working directory.',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='.',
              show_default=True)
@click.option('--vasp-exe', help='Executable for VASP', default='vasp_std', show_default=True)
@click.option('--timeout', help='Timeout in seconds to terminate VASP', default=10, show_default=True)
@click.option('--work-dir', help='Working directory for running', show_default=True)
@click.option('--keep', help='Wether to the dryrun files', is_flag=True, show_default=True)
def cmd_vasp_dryrun(input_dir, vasp_exe, timeout, work_dir, keep):
    """
    A simple tool to dryrun a VASP calculation. The calculation will be run for
    up to <timeout> seconds. The underlying VASP process will be terminated once it enters
    the main loop, which is signalled by the appearance of a `INWAV` keyword in the OUTCAR.
    """
    result = vasp_dryrun(input_dir=input_dir, vasp_exe=vasp_exe, timeout=timeout, work_dir=work_dir, keep=keep)
    with open(Path(input_dir) / 'dryrun.yaml', 'w') as fhandle:
        yaml.dump(result, fhandle, Dumper=yaml.SafeDumper)


def vasp_dryrun(input_dir, vasp_exe='vasp_std', timeout=10, work_dir=None, keep=False):
    """
    Perform a "dryrun" for a VASP calculation - get the number of kpoints, bands and
    estimated memory usage.
    """
    if not work_dir:
        tmpdir = tempfile.mkdtemp()  # tmpdir is the one to remove when finished
        work_dir = Path(tmpdir) / 'vasp_dryrun'
    else:
        work_dir = Path(work_dir) / 'vasp_dyrun'
        tmpdir = str(work_dir)
    input_dir = Path(input_dir)
    shutil.copytree(str(input_dir), str(work_dir))

    process = sb.Popen(vasp_exe, cwd=str(work_dir))
    time.sleep(0.5)  # Sleep for 5 seconds
    launch_start = time.time()
    outcar = work_dir / 'OUTCAR'
    dryrun_finish = False
    while (time.time() - launch_start < timeout) and not dryrun_finish:
        with open(outcar, 'r') as fhandle:
            for line in fhandle:
                if 'INWAV' in line:
                    dryrun_finish = True
                    break
        time.sleep(0.2)

    # Once we are out side the loop, kill VASP process
    process.kill()
    result = parse_outcar(outcar)

    if not keep:
        shutil.rmtree(tmpdir)

    return result


def parse_outcar(outcar_path):
    """
    Parse the header part of the OUTCAR

    Returns:
        A dictionary of the parsed information
    """
    output_dict = {
        'POTCARS': [],
    }
    with open(outcar_path) as fhandle:
        lines = fhandle.readlines()
    for il, line in enumerate(lines):
        if 'POTCAR:' in line:
            content = line.split(maxsplit=1)[1].strip()
            if content not in output_dict['POTCARS']:
                output_dict['POTCARS'].append(content)
        elif 'NKPTS' in line:
            tokens = line.strip().split()
            output_dict['num_kpoints'] = int(tokens[tokens.index('NKPTS') + 2])
            output_dict['num_bands'] = int(tokens[-1])
        elif 'NGX =' in line:
            tokens = line.strip().split()
            output_dict['NGX'] = int(tokens[tokens.index('NGX') + 2])
            output_dict['NGY'] = int(tokens[tokens.index('NGY') + 2])
            output_dict['NGZ'] = int(tokens[tokens.index('NGZ') + 2])
        elif 'k-points in reciprocal lattice and weights:' in line:
            kblock = lines[il + 1:il + 1 + output_dict['num_kpoints']]
            k_list = [[float(token) for token in subline.strip().split()] for subline in kblock]
            output_dict['kpoints_and_weights'] = k_list
        elif 'maximum and minimum number of plane-waves per node :' in line:
            output_dict['plane_waves_min_max'] = [float(token) for token in line.split()[-2:]]
        elif 'total amount of memory used by VASP MPI-rank0' in line:
            output_dict['max_ram_rank0'] = float(line.split()[-2])
            for subline in lines[il + 3:il + 9]:
                tokens = subline.replace(':', '').split()
                output_dict['mem_' + tokens[0]] = float(tokens[-2])
    return output_dict
