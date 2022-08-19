"""
hiphive related tools
"""
from aiida.engine import calcfunction
from aiida import orm

from ase.build.supercells import make_supercell
from ase.build import sort
from hiphive.structure_generation import generate_mc_rattled_structures


@calcfunction
def generate_rattle(prim, n_structures, cell_size, rattle_std, min_dist, **kwargs):
    """Generate rattled structure using MC"""
    atoms_ideal = sort(make_supercell(prim.get_ase(), cell_size.get_list()))  # supercell reference structure
    if 'seed' in kwargs:
        seed = kwargs['seed'].value
    else:
        seed = 42
    structures = generate_mc_rattled_structures(atoms_ideal, n_structures.value, rattle_std.value, min_dist.value, seed=seed)
    out = {}
    for i, struct in enumerate(structures):
        s = orm.StructureData(ase=struct)
        out[f'structure_{i:02d}'] = s
    out[f'ideal_structure'] = orm.StructureData(ase=atoms_ideal)
    return out
