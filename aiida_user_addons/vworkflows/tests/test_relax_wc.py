"""
Test for the relaxation workchain.

"""
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,unused-argument,redefined-outer-name,no-member, import-outside-toplevel

import numpy as np
import pytest
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import StructureData
from aiida_vasp.utils.aiida_utils import create_authinfo, get_data_node
from aiida_vasp.utils.fixtures import *
from aiida_vasp.utils.fixtures.data import (
    POTCAR_FAMILY_NAME,
    POTCAR_MAP,
)
from pymatgen.core import Structure

from aiida_user_addons.vworkflows.tests.common import data_path


def si_structure():
    """
    Setup a silicon structure in a displaced FCC setting
    """
    from aiida.plugins import DataFactory

    structure_data = DataFactory("structure")
    alat = 3.9
    lattice = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]) * alat
    structure = structure_data(cell=lattice)
    positions = [[0.1, 0.0, 0.0]]
    for pos_direct in positions:
        pos_cartesian = np.dot(pos_direct, lattice)
        structure.append_atom(position=pos_cartesian, symbols="Si")
    return structure


def setup_relax_workchain(structure, incar, opts, nkpts):
    """
    Setup the inputs for a VaspWorkChain.
    """
    from aiida.orm import Code

    from aiida_user_addons.vworkflows.relax import RelaxOptions

    inputs = AttributeDict()
    inputs.vasp = AttributeDict()
    inputs.structure = structure
    vasp = inputs.vasp
    vasp.parameters = get_data_node("dict", dict={"incar": incar})

    kpoints = get_data_node("array.kpoints")
    kpoints.set_kpoints_mesh((nkpts, nkpts, nkpts))
    vasp.kpoints = kpoints

    vasp.potential_family = get_data_node("str", POTCAR_FAMILY_NAME)
    vasp.potential_mapping = get_data_node("dict", dict=POTCAR_MAP)
    vasp.options = get_data_node(
        "dict",
        dict={
            "withmpi": False,
            "queue_name": "None",
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
            "max_wallclock_seconds": 3600,
            "prepend_text": f"export MOCK_CODE_BASE={data_path()}",
        },
    )
    vasp.settings = get_data_node(
        "dict", dict={"parser_settings": {"add_structure": True}}
    )

    mock = Code.get_from_string("mock-vasp-strict@localhost")
    vasp.code = mock
    relax_options = RelaxOptions(**opts).to_aiida_dict()
    inputs.relax_settings = relax_options
    inputs.metadata = {
        "label": structure.label,
    }
    return inputs


INCAR_RELAX_CONTINUE = {
    "algo": "normal",
    "nelm": 200,
    "isif": 3,
    "ediff": 1e-7,
    "ibrion": 2,
}

OPT_RELAX_CONTINUE = {
    "convergence_mode": "inout",
    "steps": 5,
    "convergence_on": True,
}


def test_relax_wc_continue(fresh_aiida_env, potentials, mock_vasp_strict):
    """Test with mocked vasp code for handling electronic convergence issues"""
    from aiida.cmdline.utils.common import (
        get_calcjob_report,
        get_workchain_report,
    )
    from aiida.engine import run
    from aiida.orm import Code
    from aiida.plugins import WorkflowFactory

    workchain = WorkflowFactory("vaspu.relax")

    mock_vasp_strict.store()
    create_authinfo(computer=mock_vasp_strict.computer, store=True)

    inputs = setup_relax_workchain(
        si_structure(), INCAR_RELAX_CONTINUE, OPT_RELAX_CONTINUE, 8
    )
    results, node = run.get_node(workchain, **inputs)

    called_nodes = list(node.called)
    called_nodes.sort(key=lambda x: x.ctime)

    assert node.exit_status == 0
    assert "retrieved" in results
    assert "misc" in results
    assert "remote_folder" in results

    # Sort the called nodes by creation time
    called_nodes = list(node.called)
    called_nodes.sort(key=lambda x: x.ctime)

    assert called_nodes[0].exit_status == 0
    assert called_nodes[1].exit_status == 0
