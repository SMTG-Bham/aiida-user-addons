"""
Tool for working with Phonopy calculations
"""

import math
from itertools import product
from pathlib import Path
from warnings import warn

import aiida.orm as orm
import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Dict, StructureData
from ase import Atoms
from pymatgen.core import Structure

try:
    from aiida_vasp.parsers.content_parsers.poscar import PoscarParser
except ImportError:
    from aiida_vasp.parsers.file_parsers.poscar import PoscarParser

from aiida.common.exceptions import NotExistent
from aiida_phonopy.common.utils import get_phonopy_instance

from aiida_user_addons.tools.vasp import export_vasp_calc


def export_phonon_work(work, dst, include_potcar=False):
    """
    Export a phonopy workflow

    This function only exports the calculation. Phonopy must be used again
    to re-obtain the forces
    """
    from aiida.orm import (
        CalcFunctionNode,
        Node,
        QueryBuilder,
        StructureData,
    )
    from phonopy.file_IO import (
        write_BORN,
        write_FORCE_CONSTANTS,
        write_FORCE_SETS,
    )

    dst = Path(dst)
    dst.mkdir(exist_ok=True)

    info_file = dst / "aiida_info"
    info_content = f"Label: {work.label}\nDescription: {work.description}\nUUID: {work.uuid}\n"
    info_file.write_text(info_content)

    for triple in work.get_outgoing(link_label_filter="force_calc%").all():
        fc_calc = triple.node
        fc_folder = dst / triple.link_label
        try:
            export_vasp_calc(fc_calc, fc_folder, decompress=True, include_potcar=include_potcar)
        except Exception as error:
            print(f"Cannot export the {fc_calc} - exporting its last calculation instead. The error was {error.args}")
            export_vasp_calc(
                fc_calc.called[0],
                fc_folder,
                decompress=True,
                include_potcar=include_potcar,
            )

    nac_calc = work.get_outgoing(link_label_filter="nac_calc").first()
    if nac_calc:
        export_vasp_calc(
            nac_calc.node,
            (dst / "nac_calc"),
            decompress=True,
            include_potcar=include_potcar,
        )

    # Write POSCAR file used from creating the displacements
    q = QueryBuilder()
    q.append(Node, filters={"id": fc_calc.pk}, tag="fc_calc")
    q.append(StructureData, with_outgoing="fc_calc", tag="fc_inp_structure")
    q.append(CalcFunctionNode, with_outgoing="fc_inp_structure", tag="displacement")
    q.append(StructureData, with_outgoing="displacement", edge_filters={"label": "structure"})
    disp_structure = q.one()[0]  # Structure used to create the displacements

    # Write the poscars
    poscar_parser = PoscarParser(data=disp_structure, precision=10)
    poscar_parser.write(str(dst / "POSCAR"))

    # Write processed phonon data if avaliable
    try:
        pobj = get_phonon_obj(work, nac="auto")
    except Exception as error:
        print(f"Cannot obtained the computed phonon data - manual re-computation will be needed. The error was {error.args}")
    else:
        # phonopy_disp.yaml
        pobj.save(dst / "phonopy_disp.yaml")
        pobj.save(dst / "phonopy_params.yaml")

        # FORCE_SETS and FORCE_CONSTANTS
        write_FORCE_SETS(pobj.dataset, dst / "FORCE_SETS")
        write_FORCE_CONSTANTS(pobj.get_force_constants(), dst / "FORCE_CONSTANTS")

        # Export BORN FILE
        if nac_calc:
            write_BORN(
                pobj.primitive,
                pobj.nac_params["born"],
                pobj.nac_params["dielectric"],
                dst / "BORN",
            )


def get_phonon_obj(work, nac="auto"):
    """
    Reconstruct a phonopy object from finished/unfinished workchain.

    Useful when recomputing certain properties are needed.
    """
    force_set = work.outputs.force_sets
    if nac == "auto":
        if "nac_params" in work.outputs:
            params = {"nac_params": work.outputs.nac_params}
    elif nac:
        params = {"nac_params": nac}
    else:
        params = {}

    if "relaxed_structure" in work.outputs:
        phonon = get_phonopy_instance(work.outputs.relaxed_structure, work.outputs.phonon_setting_info, params)
    else:
        # No relaxation - use the input structure directly
        phonon = get_phonopy_instance(work.inputs.structure, work.outputs.phonon_setting_info, params)

    # Treat the magmom
    try:
        phonon.unitcell.set_magnetic_moments(work.inputs.structure.base.attributes.get("MAGMOM"))
    except (KeyError, AttributeError):
        pass

    displacements = work.outputs.phonon_setting_info["displacement_dataset"]
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array("force_sets"))
    try:
        phonon.set_force_constants(work.outputs.force_constants.get_array("force_constants"))
    except (AttributeError, KeyError, NotExistent):
        phonon.produce_force_constants()
        warn("Cannot locate force constants - producing force constants from force_sets.")
    return phonon


def mode_mapping_gamma_from_work_node(work, qstart, qfinish, qsample, band, dryrun=True):
    """
    Generate mode mapping using a work node
    """
    from aiida.orm import Float, Int, Node

    force_set = work.outputs.force_sets
    if not isinstance(band, Node):
        band = Int(band)
    if not isinstance(qstart, Node):
        qstart = Float(qstart)
    if not isinstance(qfinish, Node):
        qfinish = Float(qfinish)
    if not isinstance(qsample, Node):
        qsample = Int(qsample)

    if "relaxed_structure" in work.outputs:
        args = [
            work.outputs.relaxed_structure,
            work.outputs.phonon_setting_info,
            force_set,
            work.outputs.nac_params,
            qstart,
            qfinish,
            qsample,
            band,
        ]
    else:
        # No relaxation - use the input structure directly
        args = [
            work.inputs.structure,
            work.outputs.phonon_setting_info,
            force_set,
            work.outputs.nac_params,
            qstart,
            qfinish,
            qsample,
            band,
        ]
    if dryrun:
        kwargs = {"metadata": {"store_provenance": False}}
    else:
        kwargs = {}

    return mode_mapping_gamma(*args, **kwargs)


@calcfunction
def mode_mapping_gamma(structure, phonon_settings, force_set, nac_param, qstart, qfinish, qsamples, band):
    """Generate pushed structures at gamma point"""
    phonon = get_phonopy_instance(
        structure,
        phonon_settings,
        {"nac_params": nac_param},
    )
    displacements = phonon_settings["displacement_dataset"]
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array("force_sets"))
    phonon.produce_force_constants()

    frames = {}
    qscale = math.sqrt(len(phonon.unitcell.get_scaled_positions()))
    qpoints = []
    for i, q in enumerate(np.linspace(qstart.value, qfinish.value, qsamples.value)):
        phonon.set_modulations((1, 1, 1), [[[0, 0, 0], band.value, q * qscale, 0.0]])
        cell = phonon.get_modulated_supercells()[0]
        atoms = Atoms(positions=cell.positions, cell=cell.cell, numbers=cell.numbers, pbc=True)
        struct = StructureData(ase=atoms)
        struct.label = structure.label + f" q_{i:03d}"
        frames[f"q_{i:03d}"] = struct
        qpoints.append(q)
    frames["info"] = Dict(dict={"Q_list": qpoints, "band": band.value, "qscale": qscale})
    return frames


@calcfunction
def mode_mapping_1d(
    structure: orm.StructureData,
    phonon_settings: orm.Dict,
    force_set: orm.ArrayData,
    nac_param: orm.Dict,
    qstart: orm.Float,
    qfinish: orm.Float,
    qsamples: orm.Int,
    band: orm.Int,
    q_point: orm.List,
    supercell: orm.List,
):
    """
    Generate pushed structures at any qpoint

    Args:
        structure: the input structure
        phonon_settings: the settings of the phonopy
        force_set: computed force_set for phonopy
        nac_param: NAC parameters used for Phonopy
        qstart: The start push amplitude
        qfinish: The finish push amplitude
        qsample: Number of samples for push
        band: Id of the band at q_point to be pushed
        q_point: The qpoint at which the mode should be pushed
        sueprcell: The supercell expansion for which the mode map to be calculated.

    Returns:
        A dictionary of pushed frames and mode mapping information.

    """
    phonon = get_phonopy_instance(
        structure,
        phonon_settings,
        {"nac_params": nac_param},
    )
    displacements = phonon_settings["displacement_dataset"]
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array("force_sets"))
    phonon.produce_force_constants()

    frames = {}
    qscale = math.sqrt(len(phonon.unitcell.get_scaled_positions()))
    qpoints = []
    for i, q in enumerate(np.linspace(qstart.value, qfinish.value, qsamples.value)):
        phonon.set_modulations(supercell.get_list(), [[q_point.get_list(), band.value, q * qscale, 0.0]])
        cell = phonon.get_modulated_supercells()[0]
        atoms = Atoms(positions=cell.positions, cell=cell.cell, numbers=cell.numbers, pbc=True)
        struct = StructureData(ase=atoms)
        struct.label = structure.label + f" q_{i:03d}"
        frames[f"q_{i:03d}"] = struct
        qpoints.append(q)
    frames["info"] = Dict(dict={"Q_list": qpoints, "band": band.value, "qscale": qscale})
    return frames


@calcfunction
def mode_mapping_gamma_2d(structure, phonon_settings, force_set, nac_param, settings):
    """
    Generate pushed structures at gamma point

    :param: settings - a dictionary with "qlist1", "qlist2" and "band1", "band2"
    """
    phonon = get_phonopy_instance(
        structure,
        phonon_settings,
        {"nac_params": nac_param},
    )
    displacements = phonon_settings["displacement_dataset"]
    phonon.dataset = displacements
    phonon.set_forces(force_set.get_array("force_sets"))
    phonon.produce_force_constants()

    frames = {}
    qscale = math.sqrt(len(phonon.unitcell.get_scaled_positions()))
    qpoints = []
    qlist1 = settings["qlist1"]
    qlist2 = settings["qlist2"]
    band1, band2 = settings["band1"], settings["band2"]
    for i, (q1, q2) in enumerate(product(qlist1, qlist2)):
        phonon.set_modulations(
            (1, 1, 1),
            [
                [[0, 0, 0], band1, q1 * qscale, 0.0],
                [[0, 0, 0], band2, q2 * qscale, 0.0],
            ],
        )
        phonon.write_modulations()
        struct = StructureData(pymatgen=Structure.from_file("MPOSCAR"))
        struct.label = structure.label + f" disp_{i:03d}"
        frames[f"disp_{i:03d}"] = struct
        qpoints.append([q1, q2])
    frames["info"] = Dict(dict={"Q_list": qpoints, "band1": band1, "band2": band2, "qscale": qscale})
    return frames
