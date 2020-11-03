"""
Tool for working with Phonopy calculations
"""

from pathlib import Path
from aiida_vasp.parsers.file_parsers.poscar import PoscarParser
from aiida_phonopy.common.utils import get_phonopy_instance

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


def get_phonon_obj(work, nac=None):
    """
    Reconstruct a phonopy object from finished/unfinished workchain.

    Useful when recomputing certain properties are needed.
    """
    force_set = work.outputs.force_sets
    if nac:
        params = {'nac_params': work.outputs.nac_params if nac == 'auto' else nac}
    else:
        params = {}
    phonon = get_phonopy_instance(work.outputs.relaxed_structure, work.outputs.phonon_setting_info, params)

    # Treat the magmom
    try:
        phonon.unitcell.set_magnetic_moments(work.inputs.structure.get_attribute('MAGMOM'))
    except (KeyError, AttributeError):
        pass

    displacements = work.outputs.phonon_setting_info['displacement_dataset']
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array('force_sets'))
    phonon.produce_force_constants()
    return phonon
