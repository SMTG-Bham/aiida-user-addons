"""
Module containing the OptionHolder class
"""
from typing import Tuple, List
from aiida.common.extendeddicts import AttributeDict
from aiida.common.exceptions import InputValidationError


class Option(property):
    """
    Base class for descriptors to be used as pre-defined fields for `OptionContainer`.

    The point of using these seemingly complex descriptors is to allow the target `OptionContainer`
    class to support pre-defined properties acting as the fields to be set with the following
    functionalities:

    * Tab completion of the field name
    * Assignment time checks of the correct object types
    * Default values at a per `OptionHolder` subclass level
    * Enforcement of a field being required, e.g. no default value is avaliable.
    * Automatic type convertion where necessary

    Note that the inheritance from 'property' is need for IPython introspect to work, but the usage
    is rather differently than the actual 'property' (as decorators/factories).
    However, the instantiated objects both sever as descriptors in a similar way.
    """

    def __init__(self, docstring, default_value=None, required=False):
        """Initialise an option and passing the docstring"""
        self.__doc__ = docstring
        self.required = required
        self.default_value = default_value
        self.name = None

    def __set_name__(self, owner, name):
        """Methods for automatically setting the `name` attribute - works for python 3.6+ only"""
        self.name = name

    def __get__(self, obj, owner=None):
        """Get the stored value"""
        if obj is None:
            return self
        if self.required and self.name not in obj._opt_data:
            raise ValueError(f'Field {self.name} has not been set yet!')

        return obj._opt_data.get(self.name, self.default_value)

    def __set__(self, obj, value):
        obj._opt_data[self.name] = value

    def __delete__(self, obj):
        """Delete the option from the holder dictionary"""
        if self.name in obj._opt_data:
            del obj._opt_data[self.name]


class TypedOption(Option):
    """Class for an option that enforces a specific type"""

    target_type = bool

    def __init__(self, docstring, default_value=None, required=False, enforce_type=False):
        """Instantiate an TypedOption field"""
        super().__init__(docstring, default_value, required)
        self.enforce_type = enforce_type

    def __set__(self, obj, value):
        """Setter for setting the option"""
        if self.enforce_type:
            if isinstance(value, self.target_type):
                obj._opt_data[self.name] = value
            else:
                raise ValueError(f'{value} is not a {self.target_type} type!')
        else:
            obj._opt_data[self.name] = self.target_type(value)

    def __get__(self, obj, owner=None):
        if obj is None:
            return self

        raw_value = super().__get__(obj, owner)
        if raw_value is not None:
            return self.target_type(raw_value)
        else:
            return None


class ChoiceOption(Option):
    """Option that only allow certain values"""

    def __init__(self, docstring, choices, default_value=None, required=False):
        super().__init__(docstring, default_value, required)
        self.choices = choices

    def __set__(self, obj, value):
        """Setter that sets the field"""
        if value not in self.choices:
            raise ValueError(f'{value} is not a valid choice, choose from: {self.choices}.')
        obj._opt_data[self.name] = value


class BoolOption(TypedOption):
    """Class for an option that accepts bool values"""
    target_type = bool


class FloatOption(TypedOption):
    """Class for an option that accepts float values"""
    target_type = float


class IntOption(TypedOption):
    """Class for an option that accepts integer values"""
    target_type = int


class StringOption(TypedOption):
    """Class for an option that accepts only string values"""

    def __init__(self, docstring, default_value=None, required=False, enforce_type=True):
        """Instantiate an object, note that we enforce_type by default here."""
        super().__init__(docstring, default_value=default_value, required=required, enforce_type=enforce_type)


class OptionContainer:
    """
    Base class for a container of options
    """

    def __init__(self, **kwargs):
        """
        A holder of options

        Arguments:
            kwargs: unpack keyword arguments and set them as the attributes
        """
        self._opt_data = dict()

        # the key word arguments are interpreted as the options to be set, an we check
        # If they are valid in the first place
        self.valid_options, self.required_options = self._get_valid_and_required_options()
        for key, value in kwargs.items():
            if key in self.valid_options:
                setattr(self, key, value)
            else:
                raise ValueError(f'{key} is not a valid option for a {type(self)} instance!')

    def _get_valid_and_required_options(self) -> Tuple[list, list]:
        """
        Return a list of valid option names

        This method extracts a tuple of list of the valid option names and those that are
        marked as `required`.
        """
        options = []
        required = []
        class_dict = vars(type(self))
        for name in dir(self):
            if name.startswith('__'):
                pass
            # Check if the name exists in the class's __dict__
            if name not in class_dict:
                continue
            optobj = class_dict.get(name)

            # Check if the name is an Option
            if isinstance(optobj, Option):
                options.append(name)
                if optobj.required:
                    required.append(name)
        return options, required

    @property
    def _invalid_attributes(self) -> List[str]:
        """Any attribute store inside __dict__ is an invalid option"""
        known = ['_opt_data', 'valid_options', 'required_options']
        invalid = []
        for key in self.__dict__.keys():
            if key not in known:
                invalid.append(key)
        return invalid

    def to_dict(self, check_invalids=True) -> AttributeDict:
        """Return a python dict representation - including all fields"""
        # Check for any additional attributes
        invalid_attrs = self._invalid_attributes
        if check_invalids and invalid_attrs:
            raise ValueError(f'The following attributes are not valid options: {invalid_attrs}')

        outdict = {}
        for key in self.valid_options:
            value = getattr(self, key)
            # Note that 'None' is interpreted specially meaning that the option should not be present
            if value is not None:
                outdict[key] = value
        return AttributeDict(outdict)

    def to_aiida_dict(self):
        """Return an `aiida.orm.Dict` presentation """
        from aiida.orm import Dict
        python_dict = self.to_dict()
        return Dict(dict=python_dict)

    def __setitem__(self, key, value) -> None:
        """Set items - we call the setattr method"""
        if key not in self.valid_options:
            raise KeyError(f'{key} is not an valid option for a {type(self)} instance.')
        setattr(self, key, value)

    def to_string(self) -> str:
        """In string format"""
        return self.to_dict(check_invalids=False).__repr__().replace('AttributeDict', '')

    def __getitem__(self, key):
        """Set items - we just the getattr method"""
        return getattr(self, key)

    def __repr__(self):
        string = self.to_string()
        string = string.replace('\n', ' ')
        string = string.replace('#', '')
        string = string.strip()
        if len(string) > 60:
            string = string[:60] + '...'
        return '{}<{}>'.format(type(self).__name__, string)

    @classmethod
    def validate_dict(cls, input_dict, port=None) -> None:
        """
        Vaildate a dictionary/Dict node, this can be used as the validator for
        the Port accepting the inputs
        """
        obj = cls(**input_dict)
        all_options = list(obj.valid_options)  # A copy for all options

        # Are we receiving a Dict object?
        if not isinstance(input_dict, dict):
            input_dict = input_dict.get_dict()

        for key in input_dict.keys():
            if key not in all_options:
                raise InputValidationError(f"Key '{key}' is not a valid option")
            all_options.remove(key)

        # Check for any missing required fields
        missing = [key for key in all_options if key in obj.required_options]
        if missing:
            raise InputValidationError(f'There are missing options: {missing}')

        # Check for any problems with obj.to_dict()
        try:
            obj.to_dict()
        except ValueError as error:
            raise InputValidationError(f'Error during validation: {error.args}')

    @classmethod
    def serialise(cls, value):
        """
        Serialise a dictionary into Dict

        This method can be passed as a `serializer` key word parameter of for the `spec.input` call.
        """
        obj = cls(**value)
        return obj.to_aiida_dict()

    @classmethod
    def setup_spec(cls, spec, port_name, **kwargs) -> None:
        """Setup the spec for this input"""
        # Check if we have 'required' fields
        obj = cls()
        # The constructor is different with/without any required_options
        # If there are any required options, it does not make any sense to have a default for the port.
        if obj.required_options:
            spec.input(port_name, validator=cls.validate_dict, serializer=cls.serialise, **kwargs)
        else:
            spec.input(port_name, validator=cls.validate_dict, default=lambda: cls().to_aiida_dict(), serializer=cls.serialise, **kwargs)

    @classmethod
    def get_description(cls):
        """
        Return a string for the options of a OptionContains in a human-readable format.
        """

        obj = cls()
        template = '{:>{width_name}s}:  {:10s} \n{default:>{width_name2}}: {}'
        entries = []
        for name in obj.valid_options:
            value = getattr(obj, name)
            # Each entry is name, type, doc, default value
            entries.append([name, getattr(cls, name).__doc__, str(type(value)), value])

        max_width_name = max([len(entry[0]) for entry in entries]) + 2

        lines = []
        for entry in entries:
            lines.append(template.format(*entry, width_name=max_width_name, width_name2=max_width_name + 10, default='Default'))
        return '\n'.join(lines)


########
#
#  Old option holder code
#
# ######


def typed_field(name, types, doc, default):
    """A option of certain types, with default values"""

    def getter(self):
        output = self._get_opt(name)  # pylint: disable=protected-access
        if output is None:
            output = default
        return output

    def setter(self, value):
        # not None and wrong type - warn about it
        if (value is not None) and (not isinstance(value, types)):
            raise InputValidationError(f'Value {value} is in the wrong type for {name}, the allowed types are: {types}')
        self._set_opt(name, value)  # pylint: disable=protected-access

    def deleter(self):
        self._delete_opt(name)  # pylint: disable=protected-access

    return property(getter, setter, deleter, doc)


def required_field(name, types, doc):
    """A option of certain types, must be supplied"""

    def getter(self):
        output = self._get_opt(name)  # pylint: disable=protected-access
        if output is None:
            raise InputValidationError(f'{name} is a required field!')
        return output

    def setter(self, value):
        # not None and wrong type - warn about it
        if (value is not None) and (not isinstance(value, types)):
            raise InputValidationError(f'Value {value} is in the wrong type for {name}, the allowed types are: {types}')
        self._set_opt(name, value)  # pylint: disable=protected-access

    def deleter(self):
        self._delete_opt(name)  # pylint: disable=protected-access

    return property(getter, setter, deleter, doc)


class OptionHolder(object):
    """
    A container for holding a dictionary of options.

    Valid options can be set using the standard `obj.<option> = <value>` syntax.
    This is only a helper for getting the input dictionary of settings, and allow
    populating such dictionary with the default values. This way, all settings will
    be stored in AiiDA's database.

    Example:
        The `to_aiida_dict` method should be called for serving the options to an input
        port::

            builder.my_settings = OptionHolder(a=1).to_aiida_dict()

    Workflows may choose to enable `validator` and/or `serializor` for additional checking
    corrections.

    """

    _allowed_options = tuple()
    _allowed_empty_fields = []

    @classmethod
    def _validate_class(cls):
        if not cls._allowed_options:
            raise RuntimeError('Must have non-empty _allowed_options')

    def __init__(self, **kwargs):
        """
        A holder of options

        Arguments:
            kwargs: unpack keyword arguments and set them as the attributes
        """
        self._validate_class()
        self._opt_data = dict()

        # Set the attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_opt_dict(self):
        return self._opt_data

    def _clear_all(self):
        """Set all option to be None"""
        self._opt_data.clear()

    def _get_opt(self, value):
        """Get option"""
        return self._opt_data.get(value)

    def _set_opt(self, name, value):
        """Set option"""
        return self._opt_data.__setitem__(name, value)

    def _delete_opt(self, name):
        """delete a option"""
        self._opt_data.pop(name)

    def to_string(self):
        """In string format"""
        return self.to_dict().__repr__().replace('AttributeDict', '')

    def to_dict(self):
        """
        Return a dictionary of the options with values, exclude those with
        values equal to None
        """
        return AttributeDict({key: getattr(self, key) for key in self._allowed_options if getattr(self, key) is not None})

    @classmethod
    def validate_dict(cls, input_dict, port=None):
        """
        Vaildate a dictionary/Dict node
        """
        cls._validate_class()
        all_options = list(cls._allowed_options)
        obj = cls()

        # Are we receiving a Dict object?
        if not isinstance(input_dict, dict):
            input_dict = input_dict.get_dict()

        for key, value in input_dict.items():
            if key not in cls._allowed_options:
                raise InputValidationError(f"Key '{key}' is not a valid option")
            # Try set the property - this checks the type
            setattr(obj, key, value)
            all_options.remove(key)
        # Check if any of the keys are not set
        missing = list(filter(lambda x: x not in cls._allowed_empty_fields, all_options))
        if any(missing):
            raise InputValidationError(f'Keys: {missing} are missing')
        try:
            obj.to_dict()
        except ValueError as error:
            raise InputValidationError('Problems encouterred: {}'.format(error.args))

    def __setitem__(self, key, value):
        if key not in self._allowed_options:
            raise KeyError(f'Option {key} is not a valid one!')
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, in_dict):
        """Construct from a dictionary"""
        cls._validate_class()
        obj = cls()
        all_options = list(cls._allowed_options)
        for key, value in in_dict.items():
            if key not in cls._allowed_options:
                raise InputValidationError(f"Key '{key}' is not a valid option")
            setattr(obj, key, value)
            all_options.remove(key)
        # Warn any usage of the default values
        if all_options:
            for key in all_options:
                default = getattr(obj, key)
                print(f"Warning: populated '{key}' with default '{default}'")
        return obj

    def to_aiida_dict(self):
        """Construct a orm.Dict object"""
        from aiida.orm import Dict
        return Dict(dict=self.to_dict())

    def __repr__(self):
        string = self.to_string()
        string = string.replace('\n', ' ')
        string = string.replace('#', '')
        string = string.strip()
        if len(string) > 60:
            string = string[:60] + '...'
        return '{}<{}>'.format(type(self).__name__, string)

    @classmethod
    def serialise(cls, value):
        """Serialise a dictionary into Dict"""
        cls._validate_class()

        # If a instance of the objection is passed, call the to_aiida_dict to construct the input
        if isinstance(value, cls):
            return value.to_aiida_dict()
        # Otherwise, instantiate an object and construct the input from it
        obj = cls.from_dict(value)
        return obj.to_aiida_dict()

    @classmethod
    def setup_spec(cls, spec, port_name, **kwargs):
        """Setup the spec for this input"""
        spec.input(port_name, validator=cls.validate_dict, default=lambda: cls().to_aiida_dict(), serializer=cls.serialise, **kwargs)

    @classmethod
    def get_description(cls):
        """
        Print the options of a OptionHolder in a human-readable formatted way.
        """

        obj = cls()
        template = '{:>{width_name}s}:  {:10s} \n{type:>{width_name2}}: {} \n{default:>{width_name2}}: {}'
        entries = []
        for name in cls._allowed_options:
            value = getattr(obj, name)
            # Each entry is name, type, doc, default value
            entries.append([name, getattr(cls, name).__doc__, str(type(value)), value])

        max_width_name = max([len(entry[0]) for entry in entries]) + 2

        lines = []
        for entry in entries:
            lines.append(template.format(*entry, width_name=max_width_name, width_name2=max_width_name + 10, default='Default',
                                         type='Type'))
        return '\n'.join(lines)
