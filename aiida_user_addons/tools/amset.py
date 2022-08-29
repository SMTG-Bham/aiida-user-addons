"""
AMSET related tools
"""
from functools import wraps

import aiida.orm as orm
import numpy as np
from aiida.engine import calcfunction
from aiida.orm import Node

from aiida_user_addons.common.repository import open_compressed


def outcar_lines(func):
    """Get the content of the outcar"""

    @wraps(func)
    def _inner(*args, **kwargs):
        if isinstance(args[0], Node):
            with open_compressed(args[0].outputs.retrieved, "OUTCAR") as fh:
                lines = fh.readlines()
            new_args = list(args)
            new_args[0] = lines
            args = new_args
        return func(*args, **kwargs)

    return _inner


@outcar_lines
def elastic_tensor(lines):
    """Read the dielectric tensor in GPa"""
    for i, line in enumerate(lines):
        if "TOTAL ELASTIC MODULI" in line:
            break
    tensor_lines = lines[i + 3 : i + 9]
    tensor = np.zeros((6, 6))
    for i, line in enumerate(tensor_lines):
        tensor[i, :] = [float(token) for token in line.split()[1:]]
    return tensor / 10


@outcar_lines
def high_freq_dielectric(data):
    """Get the high frequency dielectric constant tensor"""
    for i, line in enumerate(data):
        if (
            "frequency dependent      REAL DIELECTRIC FUNCTION (independent particle, no local field effects) density-density"
            in line
        ):
            elems = [float(x) for x in data[i + 3].split()[1:]]
    # Recover the full tensor
    tensor = np.zeros((3, 3))
    tensor[0, 0] = elems[0]
    tensor[1, 1] = elems[1]
    tensor[2, 2] = elems[2]
    tensor[0, 1] = elems[3]
    tensor[1, 0] = elems[3]
    tensor[1, 2] = elems[4]
    tensor[2, 1] = elems[4]
    tensor[2, 0] = elems[5]
    tensor[0, 2] = elems[5]
    return tensor


@outcar_lines
def static_ionic_dielectric_tensor(lines):
    """
    Read the static ionic dielectric tensor

    Starting with 'MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION'
    This is usually from as DFPT calculation with ISIF=2
    """
    for i, line in enumerate(lines):
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION" in line:
            # Use the second set
            idx = i

    tensor_lines = lines[idx + 2 : idx + 5]
    # print(tensor_lines)
    tensor = np.zeros((3, 3))
    for i, line in enumerate(tensor_lines):
        tensor[i, :] = [float(token) for token in line.split()]
    return tensor


@calcfunction
def get_high_freq_dielectric_martrix(node):
    """Get the high frequency dielectric constant matrix calculated from LOPTICS"""
    values = high_freq_dielectric(node)
    out = orm.List(list=values.tolist())
    return out
