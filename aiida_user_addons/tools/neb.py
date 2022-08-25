"""
Some tools for parsing data from NEB calculations
"""
from typing import Tuple, Union, TextIO
from pathlib import Path
from typing import Union
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from parsevasp.poscar import Poscar


def _parse_force_block(lines):
    """
    Parse the block of total forces from the OUTCAR file

    :param lines: A list of lines containing lines including the TOTAL-FORCE block

    :returns: A tuple of position and forces
    """
    forces = []
    positions = []
    istart = len(lines)
    for idx, line in enumerate(lines):
        if 'TOTAL-FORCE (eV/Angst)' in line:
            istart = idx
        elif idx > istart + 1:
            if not line.startswith(' -----'):  # Still in the block
                values = list(map(float, line.split()))
                positions.append(values[:3])
                forces.append(values[3:])
            else:
                # Reached the end of the block
                break
    return positions, forces


def parse_neb_outputs(fpath_or_obj: Union[str, Path, TextIO]):  # pylint: disable=too-many-branches,too-many-statements
    """
    Scan for NEB output in the OUTCAR content

    :param fpath_or_obj: Input path or fileobj

    :returns dict: A dictionary of the parsed data
    """
    if isinstance(fpath_or_obj, (str, Path)):
        with open(fpath_or_obj, 'r') as fobj:
            lines = fobj.readlines()
    # A file-like object
    elif hasattr(fpath_or_obj, 'readlines'):
        # Reset seek
        fpath_or_obj.seek(0)
        lines = fpath_or_obj.readlines()
    else:
        raise ValueError("'fpath_or_obj' variable is not supported: {}".format(fpath_or_obj))

    vtst_data = {}
    output = {'neb_converged': False}
    output['frames'] = []

    for idx, line in enumerate(lines):
        if 'NIONS' in line:
            nions = int(line.split()[-1])

        elif 'VTST: version' in line:
            vtst_data['version'] = line.split(':')[1].strip()

        elif 'NEB: Tangent' in line:
            tangents = []
            for isub in range(idx + 2, idx + 99999):
                subline = lines[isub]
                if subline.strip():
                    tangents.append([float(tmp) for tmp in subline.split()])
                else:
                    break
            vtst_data['tangents'] = tangents
        elif 'NEB: forces' in line:
            forces = [float(tmp) for tmp in line.split()[-3:]]
            vtst_data['force_par_spring'] = forces[0]
            vtst_data['force_perp_real'] = forces[1]
            vtst_data['force_dneb'] = forces[2]
        elif 'stress matrix after NEB project' in line:
            stress = []
            for isub in range(idx + 1, idx + 4):
                stress.append([float(tmp) for tmp in lines[isub].split()])
            vtst_data['stress_matrix'] = stress
        elif 'FORCES: max atom' in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data['force_max_atom'] = forces[0]
            vtst_data['force_rms'] = forces[1]
        elif 'FORCE total and by dimension' in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data['force_total'] = forces[0]
            vtst_data['force_by_dimension'] = forces[1]
        elif 'Stress total and by dimension' in line:
            forces = [float(tmp) for tmp in line.split()[-2:]]
            vtst_data['stress_total'] = forces[0]
            vtst_data['stress_by_dimension'] = forces[1]
        elif 'OPT: skip step - force has converged' in line:
            vtst_data['neb_converged'] = True
        elif 'energy(sigma->0)' in line:
            tokens = line.split()
            vtst_data['energy_extrapolated'] = float(tokens[-1])
            vtst_data['energy_without_entropy'] = float(tokens[-4])
        elif 'free  energy   TOTEN' in line:
            vtst_data['free_energy'] = float(line.split()[-2])
        elif 'TOTAL-FORCE' in line:
            positions, forces = _parse_force_block(lines[idx:idx + nions + 10])
            vtst_data['outcar-forces'] = np.array(forces)
            vtst_data['outcar-positions'] = np.array(positions)
        elif 'direct lattice vectors' in line:
            cell = []
            for subline in lines[idx + 1:idx + 4]:
                cell.append([float(tmp) for tmp in subline.split()[:3]])
            output['outcar-cell'] = np.array(cell)

        elif 'LOOP+' in line:
            # Signals end of iteration
            vtst_data['LOOP+'] = float(line.strip().split()[-1])
            output['frames'].append(vtst_data)
            vtst_data = {}

    output['last_neb_data'] = vtst_data
    return output


def get_neb_trajectory(poscar: Union[str, Path, TextIO], outcar: Union[str, Path, TextIO]):
    """
    Obtain the trajectory of NEB optimisation

    Returns two list of atoms, one for the effective projected forces and the other for the tangential forces
    """
    if isinstance(poscar, (str, Path)):
        pparser = Poscar(file_path=poscar)
    else:
        pparser = Poscar(file_handler=poscar)
    outcar_data = parse_neb_outputs(outcar)
    poscar_data = pparser.get_dict()
    sites = poscar_data['sites']
    cell = poscar_data['unitcell']
    species = [site['specie'] for site in sites]
    positions = np.stack([site['position'] for site in sites], axis=0)

    # Construct the bast atom
    base_atoms = Atoms(symbols=species, scaled_positions=positions, cell=cell, pbc=True)

    frames = []
    tangents = []
    excluded = ['tangents', 'stress_matrix']
    for frame in outcar_data['frames']:
        new_atoms = Atoms(symbols=species, positions=frame['outcar-positions'], cell=frame.get('outcar-cell', cell), pbc=True)
        new_atoms.info.update({key: value for key, value in frame.items() if not key.startswith('outcar') and key not in excluded})
        calc = SinglePointCalculator(atoms=new_atoms,
                                     energy=frame['energy_extrapolated'],
                                     forces=frame['outcar-forces'],
                                     stress=frame['stress_matrix'])
        new_atoms.calc = calc

        tangent_atoms = new_atoms.copy()
        calc = SinglePointCalculator(atoms=tangent_atoms,
                                     energy=frame['energy_extrapolated'],
                                     forces=frame['tangents'],
                                     stress=frame['stress_matrix'])
        tangent_atoms.calc = calc
        frames.append(new_atoms)
        tangents.append(tangent_atoms)
    return frames, tangents


def fix_going_through_pbc_(atoms: Atoms, atoms_ref: Atoms) -> Tuple[Atoms, np.ndarray]:
    """
    Reorder the atoms to that of the reference.

    Only works for identical or nearly identical structures that are ordered differently.
    Returns a new `Atoms` object with order similar to that of `atoms_ref` as well as the sorting indices.
    """

    # Find distances
    acombined = atoms_ref.copy()
    acombined.extend(atoms)
    new_index = []
    # Get piece-wise MIC distances
    jidx = list(range(len(atoms), len(atoms) * 2))  # Through the structure to be fixed
    for i in range(len(atoms)):
        dists = acombined.get_distances(i, jidx, mic=True)
        # Find the index of the atom with the smallest distance
        min_idx = np.where(dists == dists.min())[0][0]
        new_index.append(min_idx)
        # Fix the displacement - ensure that it is reference to the reference atom
        # Distance from the match site in the input atoms to that of the reference atom
        disp = acombined.get_distance(i, min_idx + len(atoms), mic=True, vector=True)
        # Position of the atom using the displacement vector and the original structure in the reference atoms
        acombined.positions[min_idx] = acombined.positions[i] + disp
    assert len(set(new_index)) == len(atoms), 'The detected mapping is not unique!'
    return acombined[new_index], new_index
