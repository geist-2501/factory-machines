from typing import List, Dict


def _convert_to_key_value_list(args: List[str]) -> Dict[str, str]:

    key_values = {}
    for arg in args:
        parts = arg.split('=')
        assert len(parts) == 2
        key, value = parts
        key_values[key] = value

    return key_values
