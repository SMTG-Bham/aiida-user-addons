# Name of the override name space
# This is the namespace where raw VASP INCAR tags should reside for VaspWorkChain
from aiida.common.exceptions import InputValidationError

OVERRIDE_NAMESPACE = 'incar'


def parameters_validator(node, port=None):
    """
    Validate the parameters input by passing it through the massager
    """
    from aiida_vasp.assistant.parameters import ParametersMassage, _BASE_NAMESPACES
    if not node:
        return

    pdict = node.get_dict()
    if OVERRIDE_NAMESPACE not in pdict:
        raise InputValidationError(f'Would expect some incar tags supplied under {OVERRIDE_NAMESPACE} key!')

    accepted_namespaces = _BASE_NAMESPACES + [OVERRIDE_NAMESPACE]
    new_dict = {key: value for key, value in pdict.items() if key in accepted_namespaces}
    try:
        massager = ParametersMassage(new_dict)
    except Exception as e:
        raise InputValidationError(f'Cannot validate the input parameters - error from massasager: {e}')
