"""
Tests for the common module
"""
from aiida.orm import Dict

from aiida_user_addons.vworkflows.common import (
    site_magnetization_to_magmom,
)


def test_magmom_from_site(aiida_profile):
    """Test exacting magmom"""
    output = {
        "site_magnetization": {
            "sphere": {
                "x": {
                    "site_moment": {
                        "1": {
                            "d": 0.472,
                            "f": 0.0,
                            "p": 0.021,
                            "s": 0.011,
                            "tot": 0.505,
                        },
                        "2": {
                            "d": 2.851,
                            "f": 0.0,
                            "p": 0.008,
                            "s": 0.007,
                            "tot": 2.866,
                        },
                    },
                    "total_magnetization": {
                        "d": 13.307,
                        "f": -0.012,
                        "p": 2.148,
                        "s": 0.247,
                        "tot": 15.69,
                    },
                },
                "y": {"site_moment": {}, "total_magnetization": {}},
                "z": {"site_moment": {}, "total_magnetization": {}},
            },
            "full_cell": [15.9999942],
        }
    }

    assert site_magnetization_to_magmom(output) == [0.505, 2.866]
    assert site_magnetization_to_magmom(Dict(dict=output)) == [0.505, 2.866]
