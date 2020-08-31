"""
Module containing process functions
"""
# pylint: disable=import-outside-toplevel
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from aiida.engine import calcfunction, workfunction
from aiida.orm import StructureData


@calcfunction
def refine_symmetry(struct, symprec):
    """Symmetrize a structure using Pymetgen's interface"""
    pstruc = struct.get_pymatgen()
    ana = SpacegroupAnalyzer(pstruc, symprec=symprec.value)
    ostruc = ana.get_refined_structure()
    ostruc = StructureData(pymatgen=ostruc)
    ostruc.label = struct.label + ' REFINED'
    return ostruc


@calcfunction
def extend_magnetic_orderings(struct, moment_map):
    """
    Use pymatgen to compute all possible magnetic orderings for
    a sructure.

    Returns a collection with structures containing a MAGMOM attribute
    for the per-site magnetisations.
    """
    from pymatgen.analysis.magnetism import MagneticStructureEnumerator
    from toolchest.matgen import get_all_spins
    moment_map = moment_map.get_dict()
    pstruc = struct.get_pymatgen()
    enum = MagneticStructureEnumerator(pstruc, moment_map)
    structs = {}
    for idx, ptemp in enumerate(enum.ordered_structures):
        magmom = get_all_spins(ptemp)
        for site in ptemp.sites:
            # This is abit hacky - I set the specie to be the element
            # This avoids AiiDA added addition Kind to reflect the spins
            site.species = site.species.elements[0].name
        astruc = StructureData(pymatgen=ptemp)
        astruc.set_attribute('MAGMOM', magmom)
        structs[f'out_structure_{idx:03d}'] = astruc
    return structs
