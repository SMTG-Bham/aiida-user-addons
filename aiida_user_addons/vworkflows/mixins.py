"""
Some convenience mixins
"""
from aiida.common.utils import classproperty


# pylint: disable=import-outside-toplevel, no-self-use
class WithVaspInputSet:
    """
    Mixins to attach a class property `vasp_inputset` which is the
    `VasdpInputSet` class that is used for building the input files for VASP.
    """

    @classproperty
    def vasp_inputset(self):
        from aiida_user_addons.common.inputset.vaspsets import VASPInputSet
        return VASPInputSet
