"""
Tool for working with Phonopy calculations
"""

from pathlib import Path
from aiida_vasp.parsers.file_parsers.poscar import PoscarParser

from .vasp import export_vasp_calc


def export_phonon_work(work, dst, include_potcar=False):
    """
    Export a phonopy workflow

    This function only exports the calculation. Phonopy must be used again
    to re-obtain the forces
    """

    dst = Path(dst)
    dst.mkdir(exist_ok=True)

    for triple in work.get_outgoing(link_label_filter='force_calc%').all():
        node = triple.node
        fc_folder = (dst / triple.link_label)
        export_vasp_calc(node, fc_folder, decompress=True, include_potcar=include_potcar)

    # Write POSCAR file used from creating the displacements
    if 'relax_structure' in work.outputs:
        poscar_parser = PoscarParser(data=work.outputs.relax_structure, precision=10)
    else:
        # In this case the initial structure was used as it is
        poscar_parser = PoscarParser(data=work.inputs.structure, precision=10)

    poscar_parser.write(str(dst / 'POSCAR'))
