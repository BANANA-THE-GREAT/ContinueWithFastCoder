#!/usr/bin/env python3
# coding: utf-8
"""
Module `chatette.utils`
Contains utility functions and classes used everywhere in the project.
"""

from __future__ import print_function
import sys
from random import sample, choice
from copy import deepcopy

from string import ascii_letters

from enum import Enum


class UnitType(Enum):
    alias = "alias"
    slot = "slot"
    intent = "intent"


class Singleton(object):
    """
    The base class for all singleton objects.
    Every class that subclasses this class will have the behavior
    of a singleton: their constructor will always return the same instance.
    @pre: In order to work, a sub-class needs to have an `_instance` class
          variable.
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_or_create(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    @classmethod










    @classmethod
    def was_instantiated(cls):
        return (cls._instance is not None)




























































def remove_duplicates(dict_of_lists):
    """Removes duplicates from a dictionary containing lists."""
    return {key: list(set(value)) for (key, value) in dict_of_lists.items()}















def random_string(length=6):
    """
    Returns a random string of length `length` containing only ASCII letters.
    """
    return ''.join([choice(ascii_letters) for _ in range(length)])

























if __name__ == "__main__":
    # pylint: disable=wrong-import-position
    import warnings

    warnings.warn(
        "You are running the wrong file ('utils.py')." +
        "The file that should be run is 'run.py'."
    )
