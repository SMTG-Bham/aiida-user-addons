"""
Test the utilties
"""
import pytest
from aiida.common.exceptions import InputValidationError
from .opthold import (IntOption, OptionContainer, OptionHolder, typed_field, required_field, FloatOption, BoolOption, TypedOption, Option,
                      OptionContainer)

##### Tests for the new option holder interface using descriptors


class DummyOptionClass:

    a = Option('test-option-a')
    b = Option('test-option-b', 0)
    c = Option('test-option-c', 0, True)

    def __init__(self):

        self._opt_data = {}


class DummyOptionClassWithType(DummyOptionClass):

    a = BoolOption('test option')
    b = BoolOption('test option', default_value=True, enforce_type=True)
    c = BoolOption('test option', required=True)
    d = IntOption('test option', default_value=3, enforce_type=True)
    e = FloatOption('test option', default_value=2, enforce_type=False)


def test_dummy_option_class():
    """Test for the dummy option class"""

    obj = DummyOptionClass()

    # Check for the accessor methods
    assert obj.a == None
    assert obj.b == 0
    with pytest.raises(ValueError):
        _ = obj.c

    obj.b = 10
    assert obj.b == 10


def test_dummy_option_class_with_type():
    """Test for the dummy option class with typed options"""

    obj = DummyOptionClassWithType()

    assert obj.a == False
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
        obj.e = 'abc'

    obj.e = '10.2'
    assert obj.e == 10.2


class DummyContainer(OptionContainer):

    a = FloatOption('test', 2.0)
    b = FloatOption('test', 2.0, required=True)


def test_option_container():
    """Test the option container"""

    cont = DummyContainer()
    assert cont.valid_options == ['a', 'b']
    assert cont.required_options == ['b']
    assert cont.a == 2.0

    # This should raise an error as 'b' has not been set yet
    with pytest.raises(ValueError, match='has not been set'):
        cont.to_dict()

    # Test input validation
    assert DummyContainer.validate_dict({'a': 3, 'b': 2.3}) is None

    with pytest.raises(InputValidationError, match='There are missing options'):
        DummyContainer.validate_dict({'a': 3})

    with pytest.raises(ValueError, match='c is not'):
        DummyContainer.validate_dict({'a': 3, 'b': 2.3, 'c': 2.0})

    indict = {'a': 3, 'b': 2.3}
    output = DummyContainer.serialise(indict)
    assert output.get_dict() == indict

    # Test catching invalid attribute
    cont.c = 3.0
    with pytest.raises(ValueError, match='not valid options'):
        cont.to_dict()


##### Existing tests


class DummyOptions(OptionHolder):
    _allowed_options = ('a', 'b', 'c', 'd')
    a = typed_field('a', (int,), 'a', 1)
    b = typed_field('b', (int,), 'b', 2)
    c = typed_field('c', (float,), 'c', 2.0)
    d = typed_field('d', (str,), 'd', 'foo')


class DummyOptions2(OptionHolder):
    _allowed_options = ('a', 'b', 'c', 'd')
    a = typed_field('a', (int,), 'a', 1)
    b = typed_field('b', (int,), 'b', 2)
    c = typed_field('c', (float,), 'c', 2.0)
    d = typed_field('d', (int,), 'd', 2.0)


class DummyOptions3(OptionHolder):
    a = typed_field('a', (int,), 'a', 1)
    b = typed_field('b', (int,), 'b', 2)
    c = typed_field('c', (float,), 'c', 2.0)
    d = typed_field('d', (int,), 'd', 2.0)


class DummyOptions4(OptionHolder):
    _allowed_options = (
        'a',
        'b',
        'c',
    )
    _allow_empty_field = False
    a = typed_field('a', (int,), 'a', 1)
    b = typed_field('b', (int,), 'b', 2)
    c = typed_field('c', (float,), 'c', None)


def test_constructor():
    """Test constructor of the options"""
    opts = DummyOptions(a=2, b=2, c=3.0, d='bar')
    assert opts.a == 2
    assert opts.b == 2
    assert opts.c == 3.0
    assert opts.d == 'bar'

    with pytest.raises(InputValidationError):
        opts = DummyOptions(a=2, b=2, c=3, d='bar')


def test_empty_field():
    """Test if empty field will be catached"""

    with pytest.raises(InputValidationError):
        DummyOptions4.validate_dict(dict(a=2, b=2))
    DummyOptions2.validate_dict(dict(a=2, b=2, c=3.0, d=4))


@pytest.fixture
def opts():
    return DummyOptions()


def test_to_dict(opts):
    """Test the to_dict function"""
    d = opts.to_dict()
    assert d == {'a': 1, 'b': 2, 'c': 2.0, 'd': 'foo'}


def test_validate(opts):
    """Test validating a dictionary of the options"""
    opts.d = 'bar'
    d = opts.to_dict()

    assert DummyOptions.validate_dict(d) is None
    with pytest.raises(InputValidationError):
        DummyOptions2.validate_dict(d)


def test_validate_class():
    """Test validating the class - safeguard for incorrect definition"""
    with pytest.raises(RuntimeError):
        DummyOptions3()
