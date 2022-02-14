"""
Convenient functions for surfaxe
"""

from aiida.orm import StructureData
from pymatgen.io.ase import AseAtomsAdaptor
from aiida.engine import calcfunction


@calcfunction
def surfaxes_generate_slabs(structure, hkl, thicknesses, vacuums, **other_kwargs):
    """Use surfaxe to generate slabs"""
    from surfaxe.generation import generate_slabs

    ps = structure.get_pymatgen()
    hkl = [tuple(x) for x in hkl.get_list()]
    thicknesses = thicknesses.get_list()
    vacuums = vacuums.get_list()
    additional_kwargs = other_kwargs.get('additional_kwargs', {})
    kwargs = {'save_slabs': False, **additional_kwargs}

    slabs = generate_slabs(ps, hkl, thicknesses, vacuums, **kwargs)
    print(len(slabs))
    outdict = {}
    for i, slab in enumerate(slabs):
        struct = slab.pop('slab')
        struct.remove_oxidation_states()
        struct = StructureData(ase=AseAtomsAdaptor.get_atoms(struct))
        struct.set_attribute('slab_metadata', slab)
        outdict['slab_{:02d}'.format(i)] = struct
    return outdict
