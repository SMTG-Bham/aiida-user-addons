import pytest
from ase.build import bulk

from .base import InputSet
from .castepsets import CASTEPInputSet
from .vaspsets import VASPInputSet


@pytest.fixture
def fe_atoms():
    return bulk("Fe", "fcc", 5.0)


@pytest.fixture
def mgo_atoms():
    return bulk("MgO", "rocksalt", 5.0)


def test_base(fe_atoms):
    iset = InputSet("MITRelaxSet", fe_atoms, overrides={"ediff": 1.0, "nsw": None})

    out = iset.get_input_dict()
    assert out["ediff"] == 1.0
    assert out["ibrion"] == 2
    assert "nsw" not in out


def test_vasp(fe_atoms):
    iset = VASPInputSet(
        "MITRelaxSet", fe_atoms, overrides={"ediff": 1.0, "nsw": None, "ldautype": 3}
    )

    out = iset.get_input_dict()
    assert out["ediff"] == 1.0
    assert out["ibrion"] == 2
    assert out["magmom"] == [5]
    assert out["ldauu"] == [4.0]
    assert out["ldauj"] == [0.0]
    assert out["ldaul"] == [2]
    assert out["ldautype"] == 3
    assert out["ldau"] is True
    assert "nsw" not in out


def test_kpoints(fe_atoms):
    """Test generating kpoints"""
    try:
        from aiida import load_profile
    except ImportError:
        pytest.skip("Skipping tests that need AiiDA")
    load_profile()
    inset = VASPInputSet("MITRelaxSet", fe_atoms)
    kpoints = inset.get_kpoints(0.05)
    assert kpoints.get_kpoints_mesh()[0][0] == 7


def test_castep(fe_atoms, mgo_atoms):
    """Test setting CASTEP inputs"""
    iset = CASTEPInputSet("UCLCASTEPRelaxSet", fe_atoms)

    out = iset.get_input_dict()
    assert out["task"] == "geometryoptimisation"
    assert out["smearing_width"] == 0.05

    assert out["hubbard_u"] == ["Fe  d:4.0"]

    # Check spin overrides
    iset = CASTEPInputSet(
        "UCLCASTEPRelaxSet",
        fe_atoms,
        overrides={"spin_list": [1], "hubbardu_mapping": {"Fe": ["d", 5.0]}},
    )
    out = iset.get_input_dict()
    settings = iset.get_settings()
    assert out["hubbard_u"] == ["Fe  d:5.0"]
    assert settings["SPINS"] == [1]

    # Check assigning hubbard U and spin using a mapping
    iset = CASTEPInputSet(
        "UCLCASTEPRelaxSet",
        fe_atoms,
        overrides={"spin_mapping": {"Fe": 4.0}, "hubbardu_mapping": {"Fe": ["d", 5.0]}},
    )
    out = iset.get_input_dict()
    settings = iset.get_settings()
    assert out["hubbard_u"] == ["Fe  d:5.0"]
    assert settings["SPINS"] == [4.0]

    # Check assigning spins
    iset = CASTEPInputSet(
        "UCLCASTEPRelaxSet", mgo_atoms, overrides={"spin_mapping": {"Mg": 2.0}}
    )
    out = iset.get_input_dict(flat=False)
    settings = iset.get_settings()
    assert settings["SPINS"] == [2.0, 0.6]
    assert out["PARAM"]["spin_polarised"] is True
