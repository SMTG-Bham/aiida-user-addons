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


def site_magnetization_to_magmom(site_dict):
    """Convert site mangetization to MAGMOM used for restart"""
    if 'site_magnetization' in site_dict:
        site_dict = site_dict['site_magnetization']

    site_dict = site_dict['sphere']
    to_use = None
    for symbol in 'xyz':
        if site_dict.get(symbol) and site_dict.get(symbol, {}).get('site_moment'):
            to_use = symbol
            break
    # No avaliable site magnetization for setting MAGMOM, something is wrong
    if to_use is None:
        raise RuntimeError('No valid site-projected magnetization avaliable')
    # Ensure sorted list
    tmp = list(site_dict[to_use]['site_moment'].items())
    tmp.sort(key=lambda x: int(x[0]))
    return [entry[1]['tot'] for entry in tmp]
