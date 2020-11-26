"""
Tests for magmapping
"""

from aiida_user_addons.common.magmapping import create_additional_species, convert_to_plain_list


def test_additional_species():
    """Test creating additonal species"""
    species, mapping = create_additional_species(['Fe', 'Fe'], [5, 5])
    assert species == ['Fe', 'Fe']
    assert mapping == {'Fe': 5}

    species, mapping = create_additional_species(['Fe', 'Fe'], [5, -5])
    assert species == ['Fe1', 'Fe2']
    assert mapping == {'Fe1': 5, 'Fe2': -5}

    species, mapping = create_additional_species(['Fe', 'Fe', 'O'], [5, -5, 0])
    assert species == ['Fe1', 'Fe2', 'O']
    assert mapping == {'Fe1': 5, 'Fe2': -5, 'O': 0}

    species, mapping = create_additional_species(['Fe', 'Fe', 'O'], [5, -5, 0])
    assert species == ['Fe1', 'Fe2', 'O']
    assert mapping == {'Fe1': 5, 'Fe2': -5, 'O': 0}

    species, mapping = create_additional_species(['Fe', 'Fe', 'O', 'O'], [5, -5, 0, -1])
    assert species == ['Fe1', 'Fe2', 'O1', 'O2']
    assert mapping == {'Fe1': 5, 'Fe2': -5, 'O1': 0, 'O2': -1}

    species, mapping = create_additional_species(['Fe', 'Fe', 'O', 'O', 'S', 'S'], [5, 5, 0, -1, -2, -3])
    assert species == ['Fe', 'Fe', 'O1', 'O2', 'S1', 'S2']
    assert mapping == {'Fe': 5, 'O2': -1, 'O1': 0, 'S1': -2, 'S2': -3}


def test_to_plain_list():
    """Test convertion from decorated mappings to a plain list"""
    species, mapping = create_additional_species(['Fe', 'Fe', 'O'], [5, -5, 0])
    pspecies, magmom = convert_to_plain_list(species, mapping)
    assert pspecies == ['Fe', 'Fe', 'O']
    assert magmom == [5, -5, 0]
