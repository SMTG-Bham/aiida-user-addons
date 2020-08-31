"""Test workchains functionss."""
# pylint: disable=unused-import,redefined-outer-name,unused-argument,unused-wildcard-import,wildcard-import,no-member

from importlib import import_module
import pytest
from aiida.common.exceptions import MissingEntryPointError
from aiida_vasp.utils.fixtures import fresh_aiida_env
from aiida_user_addons.vworkflows.utils.workchains import compare_structures


@pytest.fixture
def structures(fresh_aiida_env):
    from aiida.orm import StructureData
    sa = StructureData()
    sa.set_cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sa.set_pbc(True)
    sa.append_atom(symbols=['Si'], position=[0.1, 0.0, 0.0])

    sb = StructureData()
    sb.set_cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sb.set_pbc(True)
    sb.append_atom(symbols=['Si'], position=[0.9, 0.0, 0.0])

    return sa, sb


def test_structure_compare(structures):
    sa, sb = structures
    comp = compare_structures(sa, sb)
    assert comp.absolute.pos[0][0] == pytest.approx(-0.2)
    assert len(comp.absolute.pos) == 1
