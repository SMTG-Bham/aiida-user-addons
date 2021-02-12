"""
Reader for SHELX file
The functions are adapted from ase.io.res module
"""
import re
from functools import namedtuple

from ase.geometry import cellpar_to_cell
from ase import Atoms

# TITL 2LFP-11212-7612-5 -0.0373 309.998985 -1.21516192E+004 16.0000 16.2594 28 (P-1) n - 1
#              0             1        2            3             4       5    6   7   8 9 10
TitlInfo = namedtuple('TitlInfo',
                      ['label', 'pressure', 'volume', 'enthalpy', 'spin', 'spin_abs', 'natoms', 'symm', 'flag1', 'flag2', 'flag3'])


def parse_titl(line):
    """Parse titl and return a TitlInfo Object"""
    tokens = line.split()[1:]
    return TitlInfo(
        label=tokens[0],
        pressure=float(tokens[1]),
        volume=float(tokens[2]),
        enthalpy=float(tokens[3]),
        spin=float(tokens[4]),
        spin_abs=float(tokens[5]),
        natoms=int(tokens[6]),
        symm=tokens[7],
        flag1=tokens[8],
        flag2=tokens[9],
        flag3=tokens[10],
    )


def read_res(lines):
    """
    Reads a res file from a string

    Args:
        lines (str): A list of lines containing Res data.

    Returns:
        System object that is used internally by Dscribe
    """
    abc = []
    ang = []
    species = []
    coords = []
    info = dict()
    coord_patt = re.compile(
        r"""(\w+)\s+
                                ([0-9]+)\s+
                                ([0-9\-\.]+)\s+
                                ([0-9\-\.]+)\s+
                                ([0-9\-\.]+)\s+
                                ([0-9\-\.]+)""", re.VERBOSE)
    line_no = 0
    title_items = []
    while line_no < len(lines):
        line = lines[line_no]
        tokens = line.split()
        if tokens:
            if tokens[0] == 'TITL':
                # Skip the TITLE line, the information is not used
                # in this package
                title_items = parse_titl(line)

            elif tokens[0] == 'CELL' and len(tokens) == 8:
                abc = [float(tok) for tok in tokens[2:5]]
                ang = [float(tok) for tok in tokens[5:8]]
            elif tokens[0] == 'SFAC':
                for atom_line in lines[line_no:]:
                    if line.strip() == 'END':
                        break
                    match = coord_patt.search(atom_line)
                    if match:
                        species.append(match.group(1))  # 1-indexed
                        xyz = match.groups()[2:5]
                        coords.append([float(c) for c in xyz])
                    line_no += 1  # Make sure the global is updated
        line_no += 1

    return title_items, Atoms(symbols=species, scaled_positions=coords, cell=cellpar_to_cell(list(abc) + list(ang)), pbc=True, info=info)


def read_stream(stream):
    """
    Read from a stream of RES file contents, and resturn a
    list of System object
    """
    lines = []
    atoms_list = []
    in_file = False
    titl_list = []
    for line in stream:
        line = line.strip()
        # Skip any empty lines
        if not line:
            continue
        if 'TITL' in line:
            if in_file is True:
                # read the current file
                titl, atoms = read_res(lines)
                titl_list.append(titl)
                atoms_list.append(atoms)
                lines = []
            in_file = True
        if in_file:
            lines.append(line.strip())
    # Reached the end parse the last file
    titl, atoms = read_res(lines)
    titl_list.append(titl)
    atoms_list.append(atoms)
    return titl_list, atoms_list
