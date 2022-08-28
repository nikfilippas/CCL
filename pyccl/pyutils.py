import numpy as np


def expand_dim(arr, ndim, pos) -> np.ndarray:
    """Expand array an arbitrary number of dimensions.

    Arguments
    ---------
    arr : :class:`numpy.ndarray`
        Input array.
    ndim : int
        Total number of output dimensions.
    pos : Dimension where the array will be placed.

    Returns
    -------
    out : (1, ..., arr, ..., 1) :class:`numpy.ndarray`
        Expanded array.
    """
    shape = (1,)*pos + arr.shape + (1,)*(ndim-pos-1)
    return arr.reshape(shape)


def get_expanded(*arrs) -> list:
    """Orthogonalize the input arrays, in the input order."""
    ndim = len(arrs)
    arrs = [expand_dim(arr, ndim, pos) for pos, arr in enumerate(arrs)]
    return arrs


def get_broadcastable(*arrs) -> list:
    """Helper to get broadcastable arrays.

    Given some input arrays representing different quantities,
    output reshaped arrays which can be broadcast together.

    Examples
    --------

    If all arrays are of size 0 or 1, they are not reshaped:

    .. code-block::

        >>> a, b = np.array(0), np.array(1)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array(0), array(1), ())

    If the arrays are already broadcastable but **don't** have equal shapes),
    they are broadcast:

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(0, 3)
        >>> a, b = a[:, None], b[None, :]
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0, 0, 0], [1, 1, 1]]), array([[0, 1, 2], [0, 1, 2]]), (2, 3))

    If some arrays have the same shape, their dimensions are expanded
    (as they are meant to represent different quantities):

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(1, 3)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0], [1]]), array([[1, 2]]), (2, 2))

    The same applies to arrays that cannot be broadcast using NumPy rules:

    .. code-block::

        >>> a, b = np.arange(0, 2), np.arange(0, 3)
        >>> (a, b), shape = get_broadcastable(a, b)
        >>> a, b, shape
        (array([[0], [1]]), array([[0, 1, 2]]), (2, 3))
    """
    if sum([arr.size < 2 for arr in arrs]) >= len(arrs) - 1:
        # Just single numbers. Allow one array to have dimensions.
        return arrs

    shapes = [arr.shape for arr in arrs]
    if len(set(shapes)) != len(shapes):
        # Some shapes are equal! While these are broadcastable,
        # we want to orthogonalize them as they are different quantities.
        return get_expanded(*arrs)

    try:
        # Try broadcasting the arrays against each other.
        np.broadcast_shapes(*[arr.shape for arr in arrs])
        return arrs
    except ValueError:
        # Expand the dimensions if numpy broadcasting doesn't work.
        return get_expanded(*arrs)


def resample_array():
    pass


def _fftlog_transform():
    pass
