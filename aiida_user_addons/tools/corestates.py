"""
Code for handling the core states
"""

from pathlib import Path
from subprocess import run
import re
from aiida_user_addons.common.repository import open_compressed


def parse_corestates(fh):
    """Parse core-states from a stream"""
    capture = False
    data = {}
    all_data = {}
    last_blank = False
    for line in fh:
        if 'the core state eigenenergies are' in line:
            capture = True
            continue
        if capture:
            # Blank line - end of block
            if not line.split():
                if data:
                    all_data[atom_number] = data
                last_blank = True
                continue
            last_blank = False
            match = re.match(r'^ *(\d+)-', line)
            # Last line was blank and no match this time - signal the end of the block
            if not match and last_blank:
                break
            if match:
                if data:
                    all_data[atom_number] = data
                atom_number = int(match.group(1))
                tokens = line.split()[1:]
                data = {}
            else:
                tokens = line.split()
            for i in range(0, len(tokens), 2):
                data[tokens[i]] = float(tokens[i + 1])

    return all_data


def load_local_stash(node, rel, local_stash_base):
    """Return path to the local file"""
    rel = Path(rel)
    remote_path = node.outputs.remote_stash.attributes['target_basepath']
    remote_rel = Path(remote_path).relative_to(Path(remote_path).parent.parent.parent)
    dst = (local_stash_base / remote_rel)
    if (dst / rel).is_file():
        return dst / rel
    else:
        # Need to download the
        dst.mkdir(exist_ok=True, parents=True)
        run(f'rsync -av --progress {node.computer.label}:{remote_path/rel} {dst / rel}', shell=True, check=True)
        return dst / rel


def parse_node(node):

    with open_compressed(node.outputs.retrieved, 'OUTCAR') as fh:
        data = parse_corestates(fh)
    return data
