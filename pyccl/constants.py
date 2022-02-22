from collections import OrderedDict
from . import ccllib as lib
from .ccllib import (
    CCL_ERROR_CLASS, CCL_ERROR_INCONSISTENT, CCL_ERROR_INTEG,
    CCL_ERROR_LINSPACE, CCL_ERROR_MEMORY, CCL_ERROR_ROOT, CCL_ERROR_SPLINE,
    CCL_ERROR_SPLINE_EV, CCL_CORR_FFTLOG, CCL_CORR_BESSEL, CCL_CORR_LGNDRE,
    CCL_CORR_GG, CCL_CORR_GL, CCL_CORR_LP, CCL_CORR_LM)


__all__ = ('CCL_ERROR_CLASS', 'CCL_ERROR_INCONSISTENT', 'CCL_ERROR_INTEG',
           'CCL_ERROR_LINSPACE', 'CCL_ERROR_MEMORY', 'CCL_ERROR_ROOT',
           'CCL_ERROR_SPLINE', 'CCL_ERROR_SPLINE_EV', 'CCL_CORR_FFTLOG',
           'CCL_CORR_BESSEL', 'CCL_CORR_LGNDRE', 'CCL_CORR_GG', 'CCL_CORR_GL',
           'CCL_CORR_LP', 'CCL_CORR_LM')


class ClassProperty(type):
    """Metaclass that implements ``property`` to ``classmethod``."""
    # TODO: in py39 decorators `classmethod` and `property can be combined

    def __init__(cls, *args, **kwargs):
        cls._maxsize = 64

    @property
    def maxsize(cls):
        return cls._maxsize

    @maxsize.setter
    def maxsize(cls, value):
        cls._maxsize = value
        while len(cls._caches) > cls.maxsize:
            # iteratively reduce size to current maxsize
            first_item = cls.first(cls._caches)
            cls._caches.pop(first_item)


class Caching(metaclass=ClassProperty):
    """Infrastructure to hold cached objects.

    Caching is used for pre-computed objects that are expensive to compute.

    Attributes:
        maxsize (``int``):
            Maximum number of caches. If the dictionary is full, new caches
            are assigned in a rolling fashion (i.e. delete the oldest).
    """
    _caches = OrderedDict()
    _enabled = True

    @classmethod
    def first(cls, dic):
        """Get first element of ``OrderedDict``."""
        return next(iter(dic))

    @classmethod
    def cache(cls, func):
        """Decorator used to cache slow parts of the code."""

        def wrapper(instance):
            if not cls._enabled:
                return func(instance)

            key = str(hash(instance))

            if key in cls._caches:
                # output object has been cached
                return cls._caches[key]
            elif len(cls._caches) < cls.maxsize:
                # output object is not cached and there is space
                pass
            elif len(cls._caches) >= cls.maxsize:
                # output object is not cached and there is no space
                first_item = cls.first(cls._caches)
                cls._caches.pop(first_item)

            out = func(instance)
            cls._caches[key] = out
            return out
        return wrapper

    @classmethod
    def enable(cls, maxsize=64):
        cls._enabled = True
        cls.maxsize = maxsize

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def toggle(cls):
        cls._enabled = not cls._enabled


class ParamStruct(object):
    """Dict-like structure with option to freeze keys and/or values.

    Parameters:
        dic (``dict``):
            Dictionary of parameters and values
        lock_params (``bool``):
            Switch to disable new parameter creation.
        lock_values (``bool``):
            Switch to make the parameter values immutable.

    Attributes:
        locked (``(bool, bool)``):
            Lock state of the object.
        """
    _locked_params = False
    _locked_values = False

    def __init__(self, dic, lock_params=False, lock_values=False):
        self._locked_state_bak = (lock_params, lock_values)  # save lock state
        self.locked = (self._locked_state_bak)
        self._setup(dic)

    def _setup(self, dic):
        self._unlock()
        for param, value in dic.items():
            # use setattr from `object` to bypass restrictions from this class
            object.__setattr__(self, param, value)
        self._lock()

    @property
    def locked(self):
        return self._locked_params, self._locked_values

    @locked.setter
    def locked(self, state):
        try:
            params, values = state
        except TypeError:
            raise ValueError(
                "ParamStruct setter state must contain both lock states.")
        else:
            self._locked_params = params
            self._locked_values = values

    def _lock(self):
        """Return object to the saved lock state."""
        self.locked = self._locked_state_bak

    def _unlock(self):
        """Unlock the object."""
        self.locked = (False, False)

    def reload(self):
        """Reload the object."""
        dic = CCLParameters.from_struct(self._name)
        self._setup(dic)

    def copy(self):
        """Return a copy of this object."""
        return ParamStruct(self.__dict__, *self.locked)

    def __setattr__(self, param, value):
        if self._locked_values and param in self._public():
            # do not allow change of value (immutable values)
            raise NotImplementedError(
                f"Values of {self._name} are immutable.")
        if self._locked_params and not hasattr(self, param):
            # do not allow insertion of parameter
            raise KeyError(
                f"CCL global parameter {param} does not exist.")
        if ("SPLINE_TYPE" in param) and (value is not None):
            # do not allow change of spline type
            raise RuntimeError(
                "CCL spline types are immutable.")
        if (param == "A_SPLINE_MAX") and (value != 1.0):
            # do not allow change of max scale factor
            raise RuntimeError(
                "A_SPLINE_MAX is fixed to 1.0 and is not mutable.")
        object.__setattr__(self, param, value)

    def __getitem__(self, param):
        return getattr(self, param)

    def __setitem__(self, param, value):
        setattr(self, param, value)

    def __repr__(self):  # pragma: no cover
        params = self.keys()
        s = "<pyccl.constants.ParamStruct>\n"
        for i, param in enumerate(params):
            s += " {" if i == 0 else "  "
            s += f"'{param}': {getattr(self, param)}"
            s += "}" if i == len(params)-1 else ",\n"
        return s

    def _public(self):
        """Access all public attributes of an instance of this class."""
        return {param: value
                for param, value in vars(self).items()
                if not param.startswith("_")}

    def keys(self):
        return self._public().keys()

    def values(self):
        return self._public().values()

    def items(self):
        return self._public().items()


class CCLParameters(object):
    """Container class with methods to manipulate ``ParamStruct`` dicts.

    Parameters:
        None (only class method functionality implemented)
    """

    @classmethod
    def get_struct(cls, name):
        """Helper to lookup C-level struct via SWIG multiple levels deep."""
        lookup = lib  # start in lib
        names = name.split(".")
        while len(names) > 0:
            # go one level deeper as long as multiple levels exist
            lookup = getattr(lookup, names.pop(0))
        return lookup

    @classmethod
    def from_struct(cls, name, constants=False):
        """Construct a ``ParamStruct`` dict containing CCL parameters
        sourced from the C-layer.

        Arguments:
            name (``str``):
                Name of the SWIG-generated function which passes the
                parameters defined at the C-layer struct.
            constants (``bool``):
                Switch to make the parameter values immutable.

        Returns:
            ``ParamStruct``:
                Frozen dictionary of default parameters.
        """
        params = {"_name": name}  # store the ccllib function name
        struct = cls.get_struct(name)
        for param in dir(struct):
            if not param.startswith("_") and param not in ["this", "thisown"]:
                params[param] = getattr(struct, param)
        return ParamStruct(params, lock_params=True, lock_values=constants)

    @classmethod
    def populate(cls, cosmo):
        """Populate the parameters of ``Cosmology.cosmo`` with those
        stored in this instance of pyccl's config:
            - ``pyccl.gsl_params``
            - ``pyccl.spline_params``

        Arguments:
            cosmo (``SWIG``: ``ccl_cosmology``):
                The ``ccl_cosmology`` struct via ``SWIG``.
        """
        from . import gsl_params, spline_params
        for param, value in gsl_params.items():
            setattr(cosmo.gsl_params, param, value)
        for param, value in spline_params.items():
            if "SPLINE_TYPE" in param:
                continue
            if param == "A_SPLINE_MAX":
                continue
            setattr(cosmo.spline_params, param, value)
