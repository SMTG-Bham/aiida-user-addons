"""
Module containing the OptionHolder class
"""
from aiida.common.extendeddicts import AttributeDict
from aiida.common.exceptions import InputValidationError


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
