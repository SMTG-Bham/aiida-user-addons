"""
Tool for working with Phonopy calculations
"""

from pathlib import Path
from warnings import warn
from aiida_vasp.parsers.file_parsers.poscar import PoscarParser
from aiida.common.exceptions import NotExistent
from aiida_phonopy.common.utils import get_phonopy_instance

from .vasp import export_vasp_calc


def export_phonon_work(work, dst, include_potcar=False):
    """
    Export a phonopy workflow

    This function only exports the calculation. Phonopy must be used again
    to re-obtain the forces
    """
    from aiida.orm import Node, QueryBuilder, StructureData, CalcFunctionNode
    from phonopy.file_IO import write_BORN, write_FORCE_CONSTANTS, write_FORCE_SETS

    dst = Path(dst)
    dst.mkdir(exist_ok=True)

    for triple in work.get_outgoing(link_label_filter='force_calc%').all():
        fc_calc = triple.node
        fc_folder = (dst / triple.link_label)
        export_vasp_calc(fc_calc, fc_folder, decompress=True, include_potcar=include_potcar)

    nac_calc = work.get_outgoing(link_label_filter='nac_calc').first().node
    if nac_calc:
        export_vasp_calc(nac_calc, (dst / 'nac_calc'), decompress=True, include_potcar=include_potcar)

    # Write POSCAR file used from creating the displacements
    q = QueryBuilder()
    q.append(Node, filters={'id': fc_calc.pk}, tag='fc_calc')
    q.append(StructureData, with_outgoing='fc_calc', tag='fc_inp_structure')
    q.append(CalcFunctionNode, with_outgoing='fc_inp_structure', tag='displacement')
    q.append(StructureData, with_outgoing='displacement', edge_filters={'label': 'structure'})
    disp_structure = q.one()[0]  # Structure used to create the displacements

    # Write the poscars
    poscar_parser = PoscarParser(data=disp_structure, precision=10)
    poscar_parser.write(str(dst / 'POSCAR'))

    # Write processed phonon data
    pobj = get_phonon_obj(work, nac='auto')

    # phonopy_disp.yaml
    pobj.save(dst / 'phonopy_disp.yaml')
    pobj.save(dst / 'phonopy_params.yaml')

    # FORCE_SETS and FORCE_CONSTANTS
    write_FORCE_SETS(pobj.dataset, dst / 'FORCE_SETS')
    write_FORCE_CONSTANTS(pobj.get_force_constants(), dst / 'FORCE_CONSTANTS')

    # Export BORN FILE
    if nac_calc:
        write_BORN(pobj.primitive, pobj.nac_params['born'], pobj.nac_params['dielectric'], dst / 'BORN')


def get_phonon_obj(work, nac='auto'):
    """
    Reconstruct a phonopy object from finished/unfinished workchain.

    Useful when recomputing certain properties are needed.
    """
    force_set = work.outputs.force_sets
    if nac == 'auto':
        if 'nac_params' in work.outputs:
            params = {'nac_params': work.outputs.nac_params}
    elif nac:
        params = {'nac_params': nac}
    else:
        params = {}

    if 'relaxed_structure' in work.outputs:
        phonon = get_phonopy_instance(work.outputs.relaxed_structure, work.outputs.phonon_setting_info, params)
    else:
        # No relaxation - use the input structure directly
        phonon = get_phonopy_instance(work.inputs.structure, work.outputs.phonon_setting_info, params)

    # Treat the magmom
    try:
        phonon.unitcell.set_magnetic_moments(work.inputs.structure.get_attribute('MAGMOM'))
    except (KeyError, AttributeError):
        pass

    displacements = work.outputs.phonon_setting_info['displacement_dataset']
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array('force_sets'))
    try:
        phonon.set_force_constants(work.outputs.force_constants.get_array('force_constants'))
    except (AttributeError, KeyError, NotExistent):
        phonon.produce_force_constants()
        warn('Cannot locate force constants - producing force constants from force_sets.')
    return phonon
