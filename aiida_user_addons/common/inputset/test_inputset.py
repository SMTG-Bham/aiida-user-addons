from .vaspsets import VASPInputSet
from .base import InputSet
from ase.build import bulk
import pytest


@pytest.fixture
def fe_atoms():
    return bulk('Fe', 'fcc', 5.0)


def test_base(fe_atoms):
    iset = InputSet('MITRelaxSet', fe_atoms, overrides={'ediff': 1.0, 'nsw': None})

    out = iset.get_input_dict()
    assert out['ediff'] == 1.0
    assert out['ibrion'] == 2
    assert 'nsw' not in out


def test_vasp(fe_atoms):
    iset = VASPInputSet('MITRelaxSet', fe_atoms, overrides={'ediff': 1.0, 'nsw': None, 'ldautype': 3})

    out = iset.get_input_dict()
    assert out['ediff'] == 1.0
    assert out['ibrion'] == 2
    assert out['magmom'] == [5]
    assert out['ldauu'] == [4.0]
    assert out['ldauj'] == [0.0]
    assert out['ldaul'] == [2]
    assert out['ldautype'] == 3
    assert out['ldau'] is True
    assert 'nsw' not in out


def test_kpoints(fe_atoms):
    """Test generating kpoints"""
    try:
        from aiida import load_profile
    except ImportError:
        pytest.skip('Skipping tests that need AiiDA')
    load_profile()
    inset = VASPInputSet('MITRelaxSet', fe_atoms)
    kpoints = inset.get_kpoints(0.05)
    assert kpoints.get_kpoints_mesh()[0][0] == 7
