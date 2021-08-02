"""
Pymatgen related tools
"""
from typing import Tuple, List
import warnings
from aiida.plugins.factories import WorkflowFactory
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.composition import get_el_sp, gcd, formula_double_format, Composition

import aiida.orm as orm
from aiida_vasp.parsers.file_parsers.potcar import MultiPotcarIo

from aiida_user_addons.common.misc import get_energy_from_misc


def load_mp_struct(mp_id):
    """
    Load Material Project structures using its api
    """
    # Query the database
    from aiida.orm import QueryBuilder, StructureData
    import pymatgen as pmg
    q = QueryBuilder()
    # For backward compatibility - also query the extras field
    q.append(StructureData, filters={'or': [{'extras.mp_id': mp_id}, {'attributes.mp_id': mp_id}]})
    exist = q.first()

    if exist:
        return exist[0]
    # No data in the db yet - import from pymatgen
    struc = pmg.get_structure_from_mp(mp_id)
    magmom = struc.site_properties.get('magmom')
    strucd = StructureData(pymatgen=struc)
    strucd.label = strucd.get_formula()
    strucd.description = 'Imported from Materials Project ID={}'.format(mp_id)
    strucd.set_attribute('mp_id', mp_id)
    if magmom:
        strucd.set_attribute('mp_magmom', magmom)
    strucd.store()
    strucd.set_extra('mp_id', mp_id)
    if magmom:
        strucd.set_extra('mp_magmom', magmom)

    return strucd


def reduce_formula_no_polyanion(sym_amt, iupac_ordering=False) -> Tuple[str, int]:
    """
    Helper method to reduce a sym_amt dict to a reduced formula and factor.
    Unlike the original pymatgen version, this function does not do any polyanion reduction

    Args:
        sym_amt (dict): {symbol: amount}.
        iupac_ordering (bool, optional): Whether to order the
            formula by the iupac "electronegativity" series, defined in
            Table VI of "Nomenclature of Inorganic Chemistry (IUPAC
            Recommendations 2005)". This ordering effectively follows
            the groups and rows of the periodic table, except the
            Lanthanides, Actanides and hydrogen. Note that polyanions
            will still be determined based on the true electronegativity of
            the elements.

    Returns:
        (reduced_formula, factor).
    """
    syms = sorted(sym_amt.keys(), key=lambda x: [get_el_sp(x).X, x])

    syms = list(filter(lambda x: abs(sym_amt[x]) > Composition.amount_tolerance, syms))

    factor = 1
    # Enforce integers for doing gcd.
    if all((int(i) == i for i in sym_amt.values())):
        factor = abs(gcd(*(int(i) for i in sym_amt.values())))

    polyanion = []

    syms = syms[:len(syms) - 2 if polyanion else len(syms)]

    if iupac_ordering:
        syms = sorted(syms, key=lambda x: [get_el_sp(x).iupac_ordering, x])

    reduced_form = []
    for s in syms:
        normamt = sym_amt[s] * 1.0 / factor
        reduced_form.append(s)
        reduced_form.append(formula_double_format(normamt))

    reduced_form = ''.join(reduced_form + polyanion)
    return reduced_form, factor


def get_entry_from_calc(calc):
    """Get a ComputedStructure entry from a given calculation/workchain"""
    misc = calc.outputs.misc
    energy = get_energy_from_misc(misc)
    in_structure = calc.inputs.structure

    # Check if there is any output structure - support for multiple interfaces
    if 'structure' in calc.outputs:
        out_structure = calc.outputs.structure
    elif 'relax' in calc.outputs:
        out_structure = calc.outputs.relax.structure
    elif 'relax__structure' in calc.outputs:
        out_structure = calc.outputs.relax__structure
    else:
        out_structure = None

    if out_structure:
        entry_structure = out_structure.get_pymatgen()
    else:
        entry_structure = in_structure.get_pymatgen()

    if calc.process_label == 'VaspCalculation':
        incar = calc.inputs.parameters.get_dict()
        pots = set(pot.functional for pot in calc.inputs.potential.values())
        if len(pots) != 1:
            raise RuntimeError('Inconsistency in POTCAR functionals! Something is very wrong...')
        pot = pots.pop()

    elif calc.process_label == 'VaspWorkChain':
        incar = calc.inputs.parameters['incar']
        pot = calc.inputs.potential_family.value
    elif calc.process_class == WorkflowFactory('vaspu.relax'):
        incar = calc.inputs.vasp.parameters['incar']
        pot = calc.inputs.vasp.potential_family.value
    elif calc.process_class == WorkflowFactory('vasp.relax'):
        incar = calc.inputs.parameters['incar']
        pot = calc.inputs.potential_family.value
    else:
        raise RuntimeError('Cannot determine calculation inputs')

    data = {
        'functional': get_functional(incar, pot),
        'umap': get_u_map(in_structure, incar.get('ldauu')),
        'calc_uuid': calc.uuid,
        'volume': entry_structure.volume
    }

    out_entry = ComputedStructureEntry(entry_structure, energy, parameters=data)
    return out_entry


def get_u_elem(struc, ldauu, elem):
    """
    Reliably get the value of U for a given element.
    Return -1 if the entry does not have the element - so compatible with any U calculations
    """
    species = MultiPotcarIo.potentials_order(struc)
    if elem in species:
        ife = species.index(elem)
        if ldauu is None:
            return 0.
        return ldauu[ife]
    return -1


def get_u_map(struc: orm.StructureData, ldauu: List[int]) -> dict:
    """
    Reliably get the value of U for all elements.
    Return -1 if the entry does not have Fe - so compatible with any U calculations
    """
    species = MultiPotcarIo.potentials_order(struc)
    mapping = {}
    for symbol in species:
        isym = species.index(symbol)
        if ldauu is None:
            mapping[symbol] = 0.0
        else:
            mapping[symbol] = ldauu[isym]
    return mapping


def get_functional(incar: dict, pot: str) -> str:
    """
    Return the name of the functional

    Args:
        incar (dict): A dictionary for setting the INCAR
        pot (str): Potential family
    """
    if incar.get('metagga'):
        return incar.get('metagga').lower()

    if pot.startswith('LDA'):
        if incar.get('gga'):
            return 'gga+ldapp'
        else:
            return 'lda'
    elif pot.startswith('PBE'):
        gga = incar.get('gga')
        hf = incar.get('lhfcalc')
        if not hf:
            if (not gga) or gga.lower() == 'pe':
                return 'pbe'
            if gga.lower() == 'ps':
                return 'pbesol'
        else:
            if (not gga) or gga.lower() == 'pe':
                if incar.get('aexx') in [0.25, None] and (incar.get('hfscreen') - 0.2 < 0.01):
                    return 'hse06'

    return 'unknown'
