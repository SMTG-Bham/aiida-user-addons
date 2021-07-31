"""
Pymatgen related tools
"""
from typing import Tuple
from pymatgen.core.composition import get_el_sp, gcd, formula_double_format, Composition


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
