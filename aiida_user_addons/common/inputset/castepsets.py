"""
Default input sets for CASTEP
"""
from copy import deepcopy

from aiida.orm import Dict, StructureData
from aiida_castep.calculations import helper
from aiida_castep.calculations.helper import CastepHelper

from .base import InputSet


class CASTEPInputSet(InputSet):
    """Input set for VASP"""

    def __init__(self, set_name, structure, overrides=None, settings=None):
        super().__init__(set_name, structure, overrides)
        self.settings = settings if settings is not None else {}

    def get_input_dict(self, raw_python=True, flat=True):
        """Get a input dictionary"""
        out_dict = super().get_input_dict(raw_python=True)

        # Check if there is any magnetic elements - if so, ensure that spin_polarised is set to True
        spin = False if "SPINS" not in self.settings else True
        spin_mapping = deepcopy(self._presets["spin_mapping"])
        spin_mapping.update(self.overrides.get("spin_mapping", {}))

        for symbol in spin_mapping:
            if symbol in self.elements:
                spin = True
                break

        if "spin_mapping" in self.overrides or "spin_list" in self.overrides:
            spin = True

        if spin is True:
            out_dict["spin_polarised"] = True

        # Build the hubbard U field
        hubbardu_map = deepcopy(self._presets["hubbardu_mapping"])
        hubbardu_map.update(self.overrides.get("hubbardu_mapping", {}))
        hubbard_u = []
        for symbol, value in hubbardu_map.items():
            if symbol in self.elements:
                # Symbol <orb>:<U>
                hubbard_u.append(f"{symbol}  {value[0]}:{value[1]}")
        if hubbard_u:
            out_dict["hubbard_u"] = hubbard_u

        # Apply overrides again over the automatically applied keys
        self.apply_overrides(out_dict)

        helper = CastepHelper()
        if flat is True:
            helper.check_dict(out_dict, allow_flat=True, auto_fix=False)
        else:
            out_dict = helper.check_dict(out_dict, allow_flat=True, auto_fix=True)

        if not raw_python:
            out_dict = Dict(dict=out_dict)
        return out_dict

    def get_settings(self):
        """
        Get the settings input for the calculations

        The SPINS field will be set according to the defined spin_mapping of the spins or
        explicitly defined "spin_list" field of the override.
        """

        # Update with overrides
        # Setup magnetic moments
        new_settings = deepcopy(self.settings)

        # Check if we are applying the spin
        spin_mapping = deepcopy(self._presets["spin_mapping"])
        spin_mapping.update(self.overrides.get("spin_mapping", {}))
        default = spin_mapping["default"]

        # Check if spin is implicitly requested
        spin = False
        for symbol in spin_mapping:
            if symbol in self.elements:
                spin = True
                break

        if "spin_mapping" in self.overrides or "spin_list" in self.overrides:
            spin = True

        # Assign the magnetic moments
        magmom = []
        if spin:
            if isinstance(self.structure, StructureData):
                for site in self.structure.sites:
                    magmom.append(spin_mapping.get(site.kind_name, default))
            else:
                for atom in self.structure:
                    magmom.append(spin_mapping.get(atom.symbol, default))
        # Apply the spin from the spin_mapping
        if "SPINS" not in new_settings:
            if magmom:
                new_settings["SPINS"] = magmom

        # If the spin is explicity defined, apply that instead
        if "spin_list" in self.overrides:
            if "spin_mapping" not in self.overrides:
                new_settings["SPINS"] = self.overrides["spin_list"]
            else:
                raise ValueError(
                    "The spin cannot be defined in the overrides as both list and mappings."
                )

        return new_settings
