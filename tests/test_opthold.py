"""
Test the utilties
"""
import pytest
from aiida.common.exceptions import InputValidationError

from aiida_user_addons.common.opthold import (
    BoolOption,
    ChoiceOption,
    FloatOption,
    IntOption,
    Option,
    OptionContainer,
)


class DummyOptionClass:

    a = Option("test-option-a")
    b = Option("test-option-b", 0)
    c = Option("test-option-c", 0, True)
    d = Option("test-option-d", None)

    def __init__(self):

        self._opt_data = {}


class DummyOptionClassWithType(DummyOptionClass):

    a = BoolOption("test option")
    b = BoolOption("test option", default_value=True, enforce_type=True)
    c = BoolOption("test option", required=True)
    d = IntOption("test option", default_value=3, enforce_type=True)
    e = FloatOption("test option", default_value=2, enforce_type=False)


class DummyOptionClassWithChoices(DummyOptionClass):

    d = ChoiceOption("Option with choices", ["a", "b"], default_value="a")


def test_dummy_option_class():
    """Test for the dummy option class"""

    obj = DummyOptionClass()

    # Check for the accessor methods
    assert obj.a == None
    assert obj.b == 0
    with pytest.raises(ValueError):
        _ = obj.c

    assert obj.d is None

    obj.b = 10
    assert obj.b == 10


def test_dummy_option_class_with_type():
    """Test for the dummy option class with typed options"""

    obj = DummyOptionClassWithType()

    assert obj.a is None
    assert obj.b == True
    with pytest.raises(ValueError):
        obj.b = 2

    with pytest.raises(ValueError):
        y = obj.c

    assert obj.d == 3

    # Raising type inconsistency errors
    with pytest.raises(ValueError):
        obj.d = 3.0

    with pytest.raises(ValueError):
        obj.e = "abc"

    obj.e = "10.2"
    assert obj.e == 10.2


def test_dummy_option_class_with_choices():
    """Tests for the ChoiceOption"""
    obj = DummyOptionClassWithChoices()
    assert obj.d == "a"

    # This would raise an error as 'z' is not allowed
    with pytest.raises(ValueError, match="not a valid choice"):
        obj.d = "z"

    obj.d = "b"
    assert obj.d == "b"


class DummyContainer(OptionContainer):

    a = FloatOption("test", 2.0)
    b = FloatOption("test", 2.0, required=True)
    e = FloatOption("test", None, required=False)


def test_option_container():
    """Test the option container"""

    cont = DummyContainer()
    assert cont.valid_options == ["a", "b", "e"]
    assert cont.required_options == ["b"]
    assert cont.a == 2.0

    # This should raise an error as 'b' has not been set yet
    with pytest.raises(ValueError, match="has not been set"):
        cont.to_dict()

    # Test input validation
    assert DummyContainer.validate_dict({"a": 3, "b": 2.3}) is None

    with pytest.raises(InputValidationError, match="There are missing options"):
        DummyContainer.validate_dict({"a": 3})

    with pytest.raises(ValueError, match="c is not"):
        DummyContainer.validate_dict({"a": 3, "b": 2.3, "c": 2.0})

    indict = {"a": 3, "b": 2.3}
    output = DummyContainer.serialise(indict)
    assert output.get_dict() == indict

    # Test catching invalid attribute
    cont.c = 3.0
    with pytest.raises(ValueError, match="not valid options"):
        cont.to_dict()

    # Test for setting/getting items
    assert cont["a"] == 2.0
    cont["b"] = 3.2
    assert cont.b == 3.2

    # Test for to_string
    del cont.c
    cont.to_string()
    cont.__repr__()

    # Test for deletion
    del cont.a
    assert cont.a == 2.0
    assert "a" not in cont._opt_data

    del cont.b
    assert "a" not in cont._opt_data
    with pytest.raises(ValueError, match="has not been set"):
        _ = cont.b
