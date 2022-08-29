"""
Tests for the build updater and DictWrapper
"""

import pytest
from aiida.manage.tests.pytest_fixtures import aiida_profile

from aiida_user_addons.common.dictwrap import DictWrapper


def test_dict_wrap(aiida_profile):
    """
    Test DictWrapper class
    """
    from aiida.orm import Dict

    init = Dict({"a": 1, "b": 2})
    wrapper = DictWrapper(init)

    assert wrapper["a"] == 1
    wrapper["c"] = 3
    assert init.get_dict().get("c") is None

    assert wrapper["c"] == 3
    assert wrapper.node["c"] == 3
    assert wrapper.is_updated
    last_node = wrapper.node

    # Have the new node stored....
    wrapper.node.store()
    wrapper["c"] = 4
    assert wrapper["c"] == 4
    assert wrapper.node["c"] == 4
    assert wrapper.node is not init
    assert wrapper.node is not last_node
    assert wrapper._stored_node is last_node
    wrapper.validate()

    # Namespace tracking
    namespace = {"port": init}
    wrapper = DictWrapper(init, namespace, "port")
    wrapper["d"] = 4
    assert namespace["port"]["d"] == 4
