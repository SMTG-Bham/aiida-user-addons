"""
Test for the sumo related code
"""
from pathlib import Path

import pytest
from aiida.orm import load_node

from aiida_user_addons.tools.sumo import read_dos_castep


@pytest.fixture
def castep_pdos_calc(fresh_aiida_env):
    """
    Test extracing PDOS from a CASTEP calculation
    """
    from aiida.tools.archive.imports import import_archive

    archive = get_data_dir() / "castep-pdos.aiida"
    # Import the archive
    import_archive(str(archive))
    uuid = "f528eed3-1c5e-46e0-8bfa-ad79ae681419"
    return load_node(uuid)


def get_data_dir():
    """Get the directory where archives are placed"""
    return Path(__file__).parent


def test_read_pdos(castep_pdos_calc):
    """Test reading PDOS from the calculation"""
    dos, pdos = read_dos_castep(
        castep_pdos_calc,
        gaussian=0.05,
        lm_orbitals={"Mn": ("d",)},
        elements={"Mn": ("d",)},
        atoms={"Mn": (0, 1, 2, 3)},
    )
    assert "Mn" in pdos
    assert "dx2" in pdos["Mn"]
    assert dos.energies[0] == pytest.approx(-66.53742681)
