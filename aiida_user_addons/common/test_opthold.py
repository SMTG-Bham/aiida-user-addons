"""
Test the utilties
"""
import pytest
from aiida.common.exceptions import InputValidationError
from .opthold import OptionHolder, typed_field, required_field


class DummyOptions(OptionHolder):
    _allowed_options = ('a', 'b', 'c', 'd')
    a = typed_field('a', (int, ), 'a', 1)
    b = typed_field('b', (int, ), 'b', 2)
    c = typed_field('c', (float, ), 'c', 2.0)
    d = typed_field('d', (str, ), 'd', 'foo')


class DummyOptions2(OptionHolder):
    _allowed_options = ('a', 'b', 'c', 'd')
    a = typed_field('a', (int, ), 'a', 1)
    b = typed_field('b', (int, ), 'b', 2)
    c = typed_field('c', (float, ), 'c', 2.0)
    d = typed_field('d', (int, ), 'd', 2.0)


class DummyOptions3(OptionHolder):
    a = typed_field('a', (int, ), 'a', 1)
    b = typed_field('b', (int, ), 'b', 2)
    c = typed_field('c', (float, ), 'c', 2.0)
    d = typed_field('d', (int, ), 'd', 2.0)


class DummyOptions4(OptionHolder):
    _allowed_options = (
        'a',
        'b',
        'c',
    )
    _allow_empty_field = False
    a = typed_field('a', (int, ), 'a', 1)
    b = typed_field('b', (int, ), 'b', 2)
    c = typed_field('c', (float, ), 'c', None)


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
    DummyOptions2.validate_dict(dict(a=2, b=2))


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
