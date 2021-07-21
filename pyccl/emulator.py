"""
This script provides the tools needed for a uniform
framework for CCL to work with emulators.
"""
from . import ccllib as lib
from .pyutils import CCLWarning
from .core import CosmologyCalculator

import warnings
import numpy as np


class Bounds(object):
    """ Operations related to the consistency of the bounds
    within which the emulator has been trained.

    Parameters:
        bounds (dict):
            Dictionary of parameters and their bounds (``vmin, vmax``).
    """

    def __init__(self, bounds):
        self.bounds = bounds
        for par, vals in self.bounds.items():
            vmin, vmax = vals
            if not vmin <= vmax:
                raise ValueError(f"Malformed bounds for parameter {par}. "
                                 "Should be [min, max].")
            if not isinstance(vals, list):
                self.bounds[par] = list(vals)

    def check_bounds(self, proposal):
        """ Check a dictionary of proposal parameters against the bounds.

        Arguments:
            proposal (dict):
                Dictionary of proposal parameters and values for the emulator.
        """
        for par, val in proposal.items():
            if par not in self.bounds:
                warnings.warn(
                    f"Unknown bounds for parameter {par}.", CCLWarning)
            vmin, vmax = self.bounds[par]
            if not (vmin <= val <= vmax):
                raise ValueError(f"Parameter {par} out of bounds "
                                 "for current emulator configuration.")


class Emulator(object):
    """ This class is used to store and access the emulator models.

    In an independent script, each emulator model which is loaded
    into memory is stored here and can be accessed without having
    to reload it. It can hold multiple models simultaneously,
    within a self-contained running script.

    * When subclassing from this class, the method ``_load`` must be
      overridden with a method that imports, loads, and returns the
      emulator.
    * You need a method that translates any CCL parameters to a set
      of values the emulator can understand. You may implement this
      via a ``_build_emu_parameters`` method, which is a setter for
      ``_param_emu_kwargs``, the dictionary passed into the emulator.
    * You may also wish to validate that the parameters are within
      the allowed range of the emulator, via a ``_validate_bounds``
      method.
    * If the emulator contains multiple models (dependent on the
      emulator's configuration)
    """
    name = 'emulator_base'
    emulators = {}

    def __init__(self):
        if not self._has_entry():
            self._set_entry()

        if not self._has_model():
            self._set_model()

        if self._reload or not self._has_config():
            self._set_config()

    def _load(self):
        # Load and return the emulator (override this)
        raise NotImplementedError(
            "You have to override the base class `_load` method "
            "in your emulator implementation.")

    def _has_entry(self):
        # Is there an entry for this emulator?
        return self.name in self.emulators

    def _has_model(self):
        # Does the required model exist?
        self._reload = self._get_config() != self._config_emu_kwargs
        if self._reload:
            return False
        else:
            entry = self._get_entry()
            return entry["model"] is not None

    def _has_config(self):
        # Is there a stored emulator configuration?
        entry = self._get_entry()
        return entry["config"] is not None

    def _has_bounds(self, key=None):
        # Are there any stored bounds?
        entry = self._get_entry()
        if key is None:
            return entry["bounds"] is not None
        else:
            if entry["bounds"] is None:
                entry["bounds"] = {}
                return False
            elif key in entry["bounds"]:
                return entry["bounds"][key] is not None
            else:
                entry["bounds"][key] = None
                return False

    def _get_entry(self):
        # Return the stored emulator entry
        entry = self.emulators[self.name]
        return entry

    def _get_model(self):
        # Return the stored emulator model
        # or `None` if it doesn't exist
        if self._has_model():
            entry = self._get_entry()
            return entry["model"]
        else:
            return None

    def _get_config(self):
        # Return the stored emulator configuration
        # or `None` if it doesn't exist
        if self._has_config():
            entry = self._get_entry()
            return entry["config"]
        else:
            return None

    def _get_bounds(self, key=None):
        # Return the stored emulator abounds
        # or `None` if they don't exist
        entry = self._get_entry()
        if self._has_bounds(key):
            if key is None:
                return entry["bounds"]
            else:
                return entry["bounds"][key]
        else:
            return None

    def _set_entry(self):
        # Create an entry for this emulator. The entry is
        # a dictionary with the emulator name containing
        # 'model', 'config', and 'bounds'.
        keys = ["model", "config", "bounds"]
        self.emulators[self.name] = dict.fromkeys(keys)

    def _set_model(self):
        # Store the model of this emulator
        emu = self._load()
        entry = self._get_entry()
        entry["model"] = emu

    def _set_config(self):
        # Store the configuration of this emulator
        entry = self._get_entry()
        entry["config"] = self._config_emu_kwargs

    def _set_bounds(self, bounds, key=None):
        # Store the bounds of this emulator
        entry = self._get_entry()
        if key is None:
            entry["bounds"] = bounds
        else:
            if entry["bounds"] is None:
                entry["bounds"] = {key: bounds}
            else:
                entry["bounds"][key] = bounds
