from collections import defaultdict

import numpy as np


# TODO Rewrite this!
def pad(array, pad_value=0) -> np.ndarray:
    dimensions = get_max_shape(array)
    result = np.full(dimensions, pad_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError:
        pass


def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:
        yield (*index, slice(len(array))), array


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]