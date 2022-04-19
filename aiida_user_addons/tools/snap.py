"""
Snapping atoms to high symmetry positions
"""

import numpy as np

def symmetry_snap(cell_tuple, symprec, tol=1e-3):
    """
    Snap the atoms to the symmetry positions detected

    Args:
        cell_tuple: A tuple of (cell, frac_positions, numbers) as input for spglib
        symprec: Precision used for symmetry detection and snapping internally

    Returns:
        cell_tuple: A tuple of (cell, frac_positions, numbers) with atoms snapped to symmetrical
    positions with machine precision.

    """
    from spglib import get_symmetry_dataset

    cell, pos, numbers = cell_tuple
    dataset = get_symmetry_dataset((cell, pos, numbers), symprec)

    rotations = dataset['rotations'] # shape (N, 3, 3)
    disp = dataset['translations'] # shape (N, 3)
    equivalent_atoms = dataset['equivalent_atoms']

    # Find equivalent atoms by symmetry operation
    equivalent_atoms_and_symmetry = find_symmetry_relationship(cell_tuple, rotations, disp, equivalent_atoms, tol=symprec)

    # Convert to cartesian space
    rot_cart = np.zeros(rotations.shape)

    nsymm = rotations.shape[0]
    recip = np.linalg.inv(cell)
    for isim in range(nsymm):
        temp = cell @ rotations[isim] @ recip
        rot_cart[isim, :, :] = temp

    # Lattice vectors
    real_vecs_symm = np.zeros((3,3))
    symm_count = np.zeros(3)
    for isym in range(nsymm):
        vecs = cell @ rot_cart[isym, :, :].T
        for i in range(3):
            for j in range(3):
                if np.all(abs(vecs[i] - cell[j]) < symprec):
                    symm_count[j] += 1
                    real_vecs_symm[j, :] += vecs[i]
                # assumed inversion symmetry for lattice as Bravais lattice should have it
                if np.all(abs(vecs[i] + cell[j]) < symprec):
                    symm_count[j] += 1
                    real_vecs_symm[j, :] -= vecs[i]

    # Apply the new lattice vectors
    real_vecs_symm = real_vecs_symm / symm_count[:, None]

    # Symmetrise positions
    symmetrized_pos = np.zeros(pos.shape)
    for isym, rot in enumerate(rotations):
        for iion, posi in enumerate(pos):
            pos_symm_i = (rot @ posi) +  disp[isym, :]
            # Displacement to the closest image
            i_equiv = equivalent_atoms_and_symmetry[iion, isym]
            pos_symm_i = pos_symm_i - np.floor(pos_symm_i - pos[i_equiv] + 0.5)
            symmetrized_pos[i_equiv, :] += pos_symm_i

    symmetrized_pos /= nsymm

    return real_vecs_symm, symmetrized_pos, numbers


def find_symmetry_relationship(cell_tuple, symm_ops, symm_disp, equivalent_idx, tol=1e-5):
    """
    Find symmetry relationship relating each atom to other

    Return a array of size (nions, nsym)
    """
    cell, pos, numbers = cell_tuple
    nsym = symm_ops.shape[0]
    nions = pos.shape[0]

    # specie, ion indexing
    equivalent_ion = np.zeros((nions, nsym), dtype=int)
    equivalent_ion[:] = -1
    for iion, posi in enumerate(pos):

        for isym, rots in enumerate(symm_ops):
            pos_sym = (rots @ posi) + symm_disp[isym]
            # Search ions that are known to be equivalent
            for jion, posj in enumerate(pos):
                if equivalent_idx[jion] != equivalent_idx[iion]:
                    # Know to be not equivalent
                    continue
                disp = posj - pos_sym
                disp = disp - np.floor(disp + 0.5)
                # ion J maps to ion I
                if np.linalg.norm(disp) < tol:
                    equivalent_ion[iion, isym] = jion
            assert equivalent_ion[iion, isym] != -1
    assert not np.any(equivalent_ion == -1)
    return equivalent_ion