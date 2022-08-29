"""
Misc codes
"""


def get_energy_from_misc(misc):
    """
    Get energy from misc output Dict/dictionary
    """
    if "energy_no_entropy" in misc["total_energies"]:
        return misc["total_energies"]["energy_no_entropy"]
    else:
        return misc["total_energies"]["energy_extrapolated"]
