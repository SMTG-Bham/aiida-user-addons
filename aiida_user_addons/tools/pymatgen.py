"""
Pymatgen related tools
"""


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
    strucd = StructureData(pymatgen=struc)
    strucd.label = strucd.get_formula()
    strucd.description = 'Imported from Materials Project ID={}'.format(mp_id)
    strucd.set_attribute('mp_id', mp_id)
    strucd.store()
    strucd.set_extra('mp_id', mp_id)

    return strucd
