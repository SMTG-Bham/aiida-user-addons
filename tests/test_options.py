"""
Test the option settings
"""
import pytest
from aiida.common.exceptions import InputValidationError


def test_validation_relax():
    """Test validation"""
    from aiida_user_addons.vworkflows.relax import RelaxOptions

    options = {
        "algo": "cg",
        "convergence_absolute": False,
        "convergence_max_iterations": 5,
        "convergence_mode": "inout",
        "convergence_on": True,
        "convergence_positions": 0.1,
        "convergence_shape_angles": 0.1,
        "convergence_shape_lengths": 0.1,
        "convergence_volume": 0.01,
        "force_cutoff": 0.03,
        "perform": True,
        "positions": True,
        "shape": True,
        "steps": 60,
        "volume": True,
    }
    assert RelaxOptions.validate_dict(options) is None
    invalid = dict(options)

    # This is OK because the default is True for convergence_on
    invalid.pop("convergence_on")
    assert RelaxOptions.validate_dict(invalid) is None

    invalid = dict(options)
    invalid.pop("force_cutoff")
    with pytest.raises(InputValidationError):
        RelaxOptions.validate_dict(invalid)

    invalid = dict(options)
    invalid["energy_cutoff"] = 0.00001
    with pytest.raises(InputValidationError):
        RelaxOptions.validate_dict(invalid)


def test_relax():
    """Test relax options"""
    from aiida_user_addons.vworkflows.relax import RelaxOptions

    opts = RelaxOptions()
    assert opts.algo == "cg"


def test_phonon():
    """Test phonon options"""
    from aiida_user_addons.vworkflows.phonon_wc import PhononSettings

    opts = PhononSettings(mesh=10)
    assert opts.magmom is None
    with pytest.raises(InputValidationError):
        assert opts.validate_dict({"mesh": 10}) is None
    opts.supercell_matrix = [2, 2, 2]
    opts.primitive_matrix = "auto"
    assert opts.validate_dict(opts.to_aiida_dict()) is None
