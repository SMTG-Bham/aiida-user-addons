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
    if settings in work.inputs:
        settings = work.inputs.settings.get_dict()
    else:
        settings = {}
    poscar_precision = settings.get('poscar_precision', 10)
    poscar_parser = PoscarParser(data=work.outputs.relax_structure(), precision=poscar_precision)
    poscar_parser.write(str(dst / 'POSCAR'))
