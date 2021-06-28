"""
Module for VASP related stuff
"""
import gzip
import shutil
import os
import tempfile
from pathlib import Path
from functools import wraps
from itertools import zip_longest

import numpy as np

from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from aiida_vasp.parsers.file_parsers.potcar import MultiPotcarIo

from aiida.repository import FileType
from sqlalchemy.sql.elements import outparam


def with_node(f):
    """Decorator to ensure that the function receives a node"""
    from aiida.orm import load_node, Node

    @wraps(f)
    def wrapped(*args, **kwargs):
        if not isinstance(args[0], Node):
            args = list(args)
            args[0] = load_node(args[0])
        return f(*args, **kwargs)

    return wrapped


def with_node_list(f):
    """Decorator to ensure that the function receives a list of nodes in its first
    positional argument"""
    from aiida.orm import load_node, Node

    @wraps(f)
    def wrapped(*args, **kwargs):
        arg0 = list(args[0])
        for idx, item in enumerate(arg0):
            if not isinstance(item, Node):
                arg0[idx] = load_node(arg0[idx])
        args = list(args)
        args[0] = arg0
        return f(*args, **kwargs)

    return wrapped


def get_magmom(structure, mapping, default=0.6):
    """
    Get the MAGMOM from a given structure with supplied mapping
    """
    mom = []
    for site in structure.sites:
        mom.append(mapping.get(site.kind_name, default))
    return mom


def export_vasp_calc(node, folder, decompress=False, include_potcar=True):
    """
    Export a AiiDA VASP calculation

    Arguments:
        node: A VaspCalculation node or VaspWorkChain node
    """
    from aiida.orm import QueryBuilder, CalcJobNode, WorkChainNode
    from aiida.common.links import LinkType
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    # Inputs
    retrieved = node.get_outgoing(link_label_filter='retrieved').one().node
    if isinstance(node, CalcJobNode):
        calcjob = node
    elif isinstance(node, WorkChainNode):
        # In this case the node is an workchain we export the
        # 'retrieved' output link and trace to its ancestor
        calcjob = retrieved.get_incoming(link_label_filter='retrieved', link_type=LinkType.CREATE).one().node
    else:
        raise RuntimeError(f'The node {node} is not a valid calculation')
    info_file = folder / ('aiida_info')
    info_content = f'Label: {calcjob.label}\nDescription: {calcjob.description}\nUUID: {calcjob.uuid}\n'
    info_file.write_text(info_content)
    # export the retrieved outputs  and the input files
    save_all_repository_objects(retrieved, folder, decompress)
    save_all_repository_objects(calcjob, folder, decompress)
    if include_potcar:
        export_pseudos(calcjob, folder)


def export_pseudos(calc_job_node, folder):
    """Save the pseudopotential file (POTCAR)"""
    pps = calc_job_node.get_incoming(link_label_filter='potential%').nested()['potential']
    multi_potcar = MultiPotcarIo.from_structure(calc_job_node.inputs.structure, pps)
    dst = str(folder / 'POTCAR')
    multi_potcar.write(dst)


def pmg_vasprun(node, parse_xml=True, parse_potcar_file=False, parse_outcar=True, **kwargs):
    """
    Attempt to parse outputs using pymatgen
    """
    from pymatgen.io.vasp import Vasprun, Outcar
    tmpf = Path(tempfile.mkdtemp())
    export_vasp_calc(node, tmpf)

    def gz_if_necessary(fname):
        if not fname.is_file():
            fname = str(fname + '.gz')
        else:
            fname = str(fname)
        return fname

    if parse_xml:
        vrun = Vasprun(gz_if_necessary(tmpf / 'vasprun.xml'), parse_potcar_file=parse_potcar_file, **kwargs)
    else:
        vrun = None
    if parse_outcar:
        outcar = Outcar(gz_if_necessary(tmpf / 'OUTCAR'))
    else:
        outcar = None

    # Clean up the temporary directory
    shutil.rmtree(tmpf)

    return vrun, outcar


def export_relax(work, dst, include_potcar=False, decompress=False):
    """
    Export a relaxation workflow

    This function exports a series of relaxation calculations in sub-folders
    """
    from aiida.orm import Node, QueryBuilder, StructureData, CalcFunctionNode, WorkChainNode
    from aiida_vasp.parsers.file_parsers.poscar import PoscarParser

    dst = Path(dst)
    dst.mkdir(exist_ok=True)
    q = QueryBuilder()
    q.append(Node, filters={'id': work.pk})
    q.append(WorkChainNode, tag='vaspwork', project=['id', '*'])
    q.order_by({'vaspwork': {'id': 'asc'}})  # Sort by ascending PK
    for index, (pk, node) in enumerate((q.iterall())):
        relax_folder = (dst / f'relax_calc_{index:03d}')
        try:
            export_vasp_calc(node, relax_folder, decompress=decompress, include_potcar=include_potcar)
        except (ValueError, AttributeError, KeyError):
            print(f'Error exporting calculation {pk}')

    # Write POSCAR file for the input
    input_structure = work.inputs.structure
    poscar_parser = PoscarParser(data=input_structure, precision=10)
    poscar_parser.write(str(dst / 'POSCAR'))

    # Write POSCAR file for the input
    try:
        out_structure = work.inputs.outputs.relax__structure
    except AttributeError:
        out_structure = work.inputs.outputs.relax.structure
    poscar_parser = PoscarParser(data=out_structure, precision=10)
    poscar_parser.write(str(dst / 'POSCAR_RELAXED'))

    # Write the info
    info_file = dst / ('aiida_info')
    info_content = f'Label: {work.label}\nDescription: {work.description}\nUUID: {work.uuid}\n'
    info_file.write_text(info_content)


def export_neb(workchain, dst):
    """Export the neb calculation"""
    from aiida_user_addons.tools.vasp import export_vasp_calc
    from aiida import orm
    energies = {key: value['energy_without_entropy'] for key, value in workchain.outputs.neb_misc['neb_data'].items()}

    # Query for the energy computed for the end structures
    q = orm.QueryBuilder()
    q.append(orm.Node, filters={'id': workchain.inputs.initial_structure.id}, tag='root')
    q.append(orm.CalcFunctionNode, with_outgoing='root', project=['attributes.function_name'])
    q.append(
        orm.StructureData,
        with_outgoing=orm.CalcFunctionNode,
        tag='relaxed',
        project=['label'],
        #edge_filters={'label': 'init_structure'},
        edge_project=['label'])
    q.append(orm.WorkflowFactory('vaspu.relax'), with_outgoing='relaxed', project=['label', 'uuid'], tag='relaxation')
    q.append(orm.Dict,
             with_incoming='relaxation',
             edge_filters={'label': 'misc'},
             project=['attributes.total_energies.energy_extrapolated'])
    q.append(orm.CalcJobNode, with_outgoing=orm.Dict, project=['*'])
    q.distinct()

    # First export the original calculation
    export_vasp_calc(workchain, dst, decompress=True, include_potcar=False)
    ends = {}
    end_id = '{:02d}'.format(len(energies) + 1)
    for _, _, _, relax_uuid, eng, calcjob, label in q.all():
        if label.startswith('init'):
            if '00' in ends:
                print('Duplicated calculation: {relax_uuid} -> {eng} vs existing {existing}'.format(relax_uuid=relax_uuid,
                                                                                                    eng=eng,
                                                                                                    existing=ends['00']))
            else:
                ends['00'] = calcjob

        elif label.startswith('final'):
            if end_id in ends:
                print('Duplicated calculation: {relax_uuid} -> {eng} vs existing {existing}'.format(relax_uuid=relax_uuid,
                                                                                                    eng=eng,
                                                                                                    existing=ends[end_id]))
            else:
                ends[end_id] = calcjob
    # Export the end point calculation
    for key, value in ends.items():
        export_vasp_calc(value, Path(dst) / key, decompress=True, include_potcar=False)


def get_kpn_density(node, kgrid):
    """
    Get tbe kpoint density in 1/AA for a given StructureData node
    Works only for orthogonal cells.
    """
    try:
        from aiida.orm import Node
    except ImportError:
        pstruc = node
    else:
        if isinstance(node, Node):
            pstruc = node.get_pymatgen()
        else:
            pstruc = node
    abc = pstruc.lattice.abc
    abc_ = 1.0 / np.array(abc) / np.asarray(kgrid)
    return np.mean(abc_), abc_


def group_diff(incars, ignored_keys=None):
    """
    Try to diff multiple incar inputs by remove keys that are identical in
    order to highly those that are different
    """
    # Ignored keys
    if ignored_keys is None:
        ignored_keys = {}

    # Make sure keys are all in lowercase
    incars = [lower_case_keys(incar) for incar in incars]

    all_keys = set()
    for incar in incars:
        for key in incar.keys():
            all_keys.add(key)
    diff_keys = set()  # Hold keys are are different
    for key in all_keys:
        if key in ignored_keys:
            continue

        vtmp = []  # Hold value of the key in all INCARS
        for incar in incars:
            vtmp.append(incar.get(key))
        # Check is vtmp are all the same
        if vtmp.count(vtmp[0]) != len(vtmp):
            # This key has different values
            diff_keys.add(key)

    comm_keys = set(key for key in all_keys if key not in diff_keys)
    # Reconstruct the list of dictionaries of key that are different amont the
    # inputs
    incar_diff = []
    for incar in incars:
        incar_diff.append({key: incar.get(key) for key in diff_keys})

    # Construct a dictionary of the key that values are all the same among input
    # dictionaries
    incar_comm = {key: value for key, value in incars[0].items() if key in comm_keys}
    return incar_diff, incar_comm


def lower_case_keys(dic):
    return {key.lower(): value for key, value in dic.items()}


@with_node_list
def compare_incars(nodes):
    ignored = ['magmom', 'ncore', 'kpar', 'npar ']
    incardicts = [node.inputs.parameters.get_dict() for node in nodes]

    return group_diff(incardicts, ignored)


def _traj_node_to_atoms(traj, energies=None):
    """Convert TrajectorData from aiida-vasp to ase.Atoms"""
    symbols = traj.get_attribute('symbols')
    cells, positions, forces = [traj.get_array(n) for n in ['cells', 'positions', 'forces']]
    atoms_list = []
    if energies is not None:
        engs = energies.get_array('energy_no_entropy')
    else:
        engs = []

    for cell, pos, forc, eng in zip_longest(cells, positions, forces, engs):
        atoms = Atoms(scaled_positions=pos, cell=cell, symbols=symbols, pbc=True)
        calc = SinglePointCalculator(atoms, forces=forc, energy=eng)
        atoms.set_calculator(calc)
        atoms_list.append(atoms)

    return atoms_list


@with_node
def traj_to_atoms(node):
    """Converge trajectory nodes to atoms"""
    from aiida.orm import CalcJobNode, WorkChainNode, TrajectoryData, Node, QueryBuilder, ArrayData
    if isinstance(node, CalcJobNode):
        traj = node.outputs.trajectory
        etmp = node.get_outgoing(link_label_filter='energies').all()
        if etmp:
            eng = etmp[0].node
        else:
            eng = None
        return _traj_node_to_atoms(traj, eng)

    if isinstance(node, WorkChainNode):

        # A better test would be if the Node has nested calls?
        if 'Relax' in node.process_label:
            # Query for all trajectory data in sub workchains called
            q = QueryBuilder()
            q.append(WorkChainNode, filters={'id': node.pk}, tag='root')
            q.append(WorkChainNode, with_incoming='root', tag='vcalc')
            q.append(CalcJobNode, tag='calc', with_incoming='vcalc')
            q.append(TrajectoryData, with_incoming='calc')
            q.order_by({'calc': ['id']})
            q.distinct()
            trajs = [tmp[0] for tmp in q.all()]

            # Query for all energies data in sub workchains called
            q = QueryBuilder()
            q.append(WorkChainNode, filters={'id': node.pk}, tag='root')
            q.append(WorkChainNode, with_incoming='root', tag='vcalc')
            q.append(CalcJobNode, tag='calc', with_incoming='vcalc')
            q.append(ArrayData, with_incoming='calc', edge_filters={'label': 'energies'})
            q.order_by({'calc': ['id']})
            q.distinct()
            engs = [tmp[0] for tmp in q.all()]
            # Note energies may not be there - but it should be handled
            atoms_lists = [_traj_node_to_atoms(traj, eng) for traj, eng in zip_longest(trajs, engs)]
            out = []
            for atom_list in atoms_lists:
                out.extend(atom_list)
            return out
        else:
            # We are dealing with a single workchain
            q = QueryBuilder()
            q.append(WorkChainNode, filters={'id': node.pk})
            q.append(Node, tag='res', with_incoming=WorkChainNode)
            q.append(CalcJobNode, tag='calc', with_outgoing='res')
            q.append(TrajectoryData, with_incoming='calc')
            q.distinct()
            traj = q.first()[0]

            q = QueryBuilder()
            q.append(WorkChainNode, filters={'id': node.pk})
            q.append(Node, tag='res', with_incoming=WorkChainNode)
            q.append(CalcJobNode, tag='calc', with_outgoing='res')
            q.append(ArrayData, edge_filters={'label': 'energies'}, with_incoming='calc')
            q.distinct()
            if q.count() > 0:
                eng = q.first()[0]
            else:
                eng = None
            return _traj_node_to_atoms(traj, eng)


def copy_from_aiida(name: str, node, dst: Path, decompress=False):
    """
    Copy objects from aiida repository.

    :param name: The full name (including the parent path) of the object.
    :param node: Node object for which the files in the repo to be copied.
    :param dst: Path of the destination folder.

    This is a recursive function so directory copying also works.
    """
    obj = node.get_object(name)

    # If it is a directory, copy the contents one by one
    if obj.file_type == FileType.DIRECTORY:
        for sub_obj in node.list_objects(name):
            copy_from_aiida(os.path.join(name, sub_obj.name), node, dst)
    else:
        # It is a file
        with node.open(name, mode='rb') as fsource:
            # Make parent directory if needed
            frepo_path = dst / name
            Path(frepo_path.parent).mkdir(exist_ok=True, parents=True)
            # Write the file
            if name.endswith('.gz') and decompress:
                out_path = str(frepo_path)[:-3]
                out_decompress = True
            else:
                out_decompress = False
                out_path = str(frepo_path)

            if not out_decompress:
                with open(out_path, 'wb') as fdst:
                    shutil.copyfileobj(fsource, fdst)
            else:
                gobj = gzip.GzipFile(fileobj=fsource, mode='rb')
                with open(out_path, 'wb') as fdst:
                    shutil.copyfileobj(gobj, fdst)


def save_all_repository_objects(node, folder: Path, decompress=False):
    """Copy all objects of a node saved in the repository to the disc"""
    for ifile in node.list_objects():
        name, _ = ifile
        copy_from_aiida(name, node, folder, decompress)
