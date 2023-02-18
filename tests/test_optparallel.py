"""
Tests for hte JobScheme
"""
import warnings

from aiida_user_addons.tools.optparallel import JobScheme, factors


def test_kpar():
    """Test setting KPAR"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        scheme = JobScheme(121, 130)
        assert scheme.kpar == 1

        scheme = JobScheme(42, 72)
        assert scheme.kpar == 6

        scheme = JobScheme(42, 72, nbands=100, npw=10000)
        assert scheme.size_wavefunction_per_proc < scheme.wf_size_limit

        # Extreme case
        scheme = JobScheme(42, 72, nbands=100, npw=10000000)
        assert scheme.kpar == 1


def test_ncore():
    """Test NCORE settings logic"""

    scheme = JobScheme(42, 72, cpus_per_node=24, npw=16000, nbands=180, ncore_strategy="balance")
    # KGROUP has 12 processes
    assert scheme.ncore == 3

    scheme = JobScheme(84, 72, cpus_per_node=24, npw=16000, nbands=180, ncore_strategy="balance")
    # KGROUP has 6 processes
    assert scheme.ncore == 2

    scheme = JobScheme(1, 72, cpus_per_node=24, npw=16000, nbands=121, ncore_strategy="balance")
    # KGROUP has 72 processes
    assert scheme.ncore == 8


def test_factors():
    """Test teh factors function"""
    assert factors(6) == [6, 3, 2, 1]
    assert factors(7) == [7, 1]
