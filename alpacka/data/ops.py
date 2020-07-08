"""Ops for manipulating pytrees.

Pytrees are nested structures of lists/tuples/namedtuples/dicts. They typically
have numpy arrays at leaves.

Behavior of those ops is well described in alpacka.data.ops_test.
"""

import numpy as np


_leaf_types = set()


def register_leaf_type(leaf_type):
    """Registers a leaf type.

    Leaf type is a type that should not be recursed into in nested_* functions.
    Useful for defining namedtuples such as data.TensorSignature.

    Args:
        leaf_type (type): Type to register as a leaf.
    """
    _leaf_types.add(leaf_type)


def _is_leaf(x):
    """Returns whether pytree is a leaf."""
    if type(x) in _leaf_types:  # pylint: disable=unidiomatic-typecheck
        return True
    return not isinstance(x, (tuple, list, dict))


def _is_namedtuple_instance(x):
    """Determines if x is an instance of a namedtuple type."""
    if isinstance(x, tuple):
        return hasattr(x, '_fields')
    else:
        return False


def nested_map(f, x, stop_fn=_is_leaf):
    """Maps a function through a pytree.

    Args:
        f (callable): Function to map.
        x (any): Pytree to map over.
        stop_fn (callable): Optional stopping condition for the recursion. By
            default, stops on leaves.

    Returns:
        pytree: Result of mapping.
    """
    if stop_fn(x):
        return f(x)

    if _is_namedtuple_instance(x):
        return type(x)(*nested_map(f, tuple(x), stop_fn=stop_fn))
    if isinstance(x, dict):
        return {k: nested_map(f, v, stop_fn=stop_fn) for (k, v) in x.items()}
    assert isinstance(x, (list, tuple)), (
        'Non-exhaustive pattern match for {}.'.format(type(x))
    )
    return type(x)(nested_map(f, y, stop_fn=stop_fn) for y in x)


def nested_zip(xs):
    """Zips a list of pytrees.

    Inverse of nested_unzip.

    Example:
        nested_zip([((1, 2), 3), ((4, 5), 6)]) == (([1, 4], [2, 5]), [3, 6])
    """
    assert not _is_leaf(xs)
    assert xs

    if _is_leaf(xs[0]):
        return xs

    if _is_namedtuple_instance(xs[0]):
        return type(xs[0])(*nested_zip([tuple(x) for x in xs]))
    elif isinstance(xs[0], (list, tuple)):
        for x in xs:
            assert len(x) == len(xs[0]), (
                'Cannot zip sequences of different lengths: '
                '{} and {}'.format(len(x), len(xs[0]))
            )

        return type(xs[0])(
            nested_zip([x[i] for x in xs]) for i in range(len(xs[0]))
        )
    elif isinstance(xs[0], dict):
        return {k: nested_zip([x[k] for x in xs]) for k in xs[0].keys()}
    else:
        raise TypeError(
            'Non-exhaustive pattern match for {}.'.format(type(xs[0]))
        )


def _is_last_level(x):
    """Returns whether pytree is at the last level (children are leaves)."""
    if _is_leaf(x):
        return False
    if isinstance(x, dict):
        vs = x.values()
    else:
        vs = x
    return all(map(_is_leaf, vs))


def _is_last_level_nonempty(x):
    """Returns whether pytree is at the last level and has any child."""
    return x and _is_last_level(x)


def nested_unzip(x):
    """Uzips a pytree of lists.

    Inverse of nested_unzip.

    Example:
        nested_unzip((([1, 4], [2, 5]), [3, 6])) == [((1, 2), 3), ((4, 5), 6)]
    """
    if not x:
        raise ValueError(f'Cannot infer expected output structure from empty '
                         f'object (got {x} of type {type(x)})')
    acc = []
    try:
        i = 0
        while True:
            acc.append(nested_map(
                lambda l: l[i],
                x,
                stop_fn=_is_last_level_nonempty,
            ))
            i += 1
    except IndexError:
        return acc


def nested_zip_with(f, xs):
    """Zips a list of pytrees using a function.

    Example (elementwise addition of 2 pytrees):
        nested_zip_with(
            lambda x, y: x + y, [((1, 2), 3), ((4, 5), 6)]
        ) == ((5, 7), 9)
    """
    def f_star(args):
        return f(*args)
    return nested_map(f_star, nested_zip(xs), stop_fn=_is_last_level_nonempty)


def nested_stack(xs):
    """Stacks a list of pytrees of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_stack([(1, 2), (3, 4)]) == (np.array([1, 3]), np.array([2, 4]))
    """
    return nested_map(np.stack, nested_zip(xs), stop_fn=_is_last_level_nonempty)


def nested_unstack(x):
    """Unstacks a pytree of numpy arrays.

    Inverse of nested_unstack.

    Example:
        nested_unstack((np.array([1, 3]), np.array([2, 4]))) == [(1, 2), (3, 4)]
    """
    def unstack(arr):
        (*slices,) = arr
        return slices
    return nested_unzip(nested_map(unstack, x))


def nested_concatenate(xs):
    """Concatenates a list of pytrees of numpy arrays."""
    return nested_map(
        np.concatenate, nested_zip(xs), stop_fn=_is_last_level_nonempty
    )


def choose_leaf(x):
    """Chooses a leaf of the pytree."""
    class Found(Exception):
        pass

    def find_leaf(x):
        raise Found(x)

    try:
        nested_map(find_leaf, x)
        raise ValueError('Pytree has no leaves.')
    except Found as e:
        (leaf,) = e.args
        return leaf
