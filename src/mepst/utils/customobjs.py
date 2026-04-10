"""
Module containing custom subclasses of Path and dict.
These subclasses have additional behaviour which is considered
to be practical by the authors.

Classes here defined are of little utility to the end user.
"""

import typing
import functools
from pathlib import (
    Path as _Path_,
    PosixPath as _PosixPath_,
    WindowsPath as _WindowsPath_,
)
import os
import argparse
from time import time

# TODO: define __all__


def first_arg_is_path(func: typing.Callable):
    """Cast the first argument to pathlib.Path"""

    @functools.wraps(func)
    def path_is_casted(*args, **kwargs):
        if not isinstance(args[0], Path):
            _casted_args = (Path(args[0]), *args[1:])
            return func(*_casted_args, **kwargs)
        else:
            return func(*args, **kwargs)

    return path_is_casted


def dont_overwite_if_exists(func: typing.Callable):
    """Prevent a function whose first argument is a pathlib.Path
    from overwriting the file if exits.
    This decorator is intended to be used together with `first_arg_is_path`"""

    @functools.wraps(func)
    def _check_before_writing(*args, **kwargs):
        if args[0].exists():
            _p = args[0].resolve().as_posix()
            raise FileExistsError(f"{func.__name__} won't overwrite file `{_p}`")
        return func(*args, **kwargs)

    return _check_before_writing


# TODO: write a generic version of this
def second_arg_is_path(func: typing.Callable):
    """Cast the second argument to pathlib.Path"""

    @functools.wraps(func)
    def path_is_casted(*args, **kwargs):
        if not isinstance(args[1], Path):
            _casted_args = args[:1], (Path(args[1]), *args[2:])
            return func(*_casted_args, **kwargs)
        else:
            return func(*args, **kwargs)

    return path_is_casted


class Timer(object):
    """A simple timer class used to measure execution time,
    without all the problems related to using timeit."""

    def __init__(self, description):
        self.description = description
        self.start: float
        self.end: float

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time()
        print(f"{self.description}: {round(self.end - self.start, 5)}", flush=True)


class Path(_Path_):
    """
    Basically a wrapper for pathlib.Path,
    created to modify pathlib.Path.glob's behaviour :

    Added a method lglob() which returns a list instead of a generator.

    Thanks to this tread on code review :
      https://codereview.stackexchange.com/questions/162426/subclassing-pathlib-path
    """

    def __new__(cls, *args, **kvps):
        return super().__new__(
            WindowsPath if os.name == "nt" else PosixPath, *args, **kvps
        )

    def lglob(self, expr: str, names_only: bool = False):
        _glob_generator = super().glob(expr)
        if names_only:
            return [f.name for f in _glob_generator]
        return list(_glob_generator)

    @property
    def abs(self):
        return super().absolute().as_posix()


class WindowsPath(_WindowsPath_, Path):
    """Helper for Path, not to be directly initialized."""


class PosixPath(_PosixPath_, Path):
    """Helper for Path, not to be directly initialized."""


class ObjDict(dict):
    """
    Instantiate a dictionary that allows accessing values
    with object notation (as if they were attributes):

    ex:
        x.foo = 5
    instead of
        x['foo'] = 5

    The best part is that both ways work !

    Ideal for working with TOML files.

    Additional features include :
        * lkeys property : same as dict.keys() but returns a list directly.
        * lvalues property : same as dict.values() but returns a list directly.

    Original code snippet found here :
        https://goodcode.io/articles/python-dict-object/
    """

    @property
    def lkeys(self):
        return list(super().keys())

    @property
    def lvalues(self):
        return list(super().values())

    @property
    def litems(self):
        return super().items()

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class StoreDictKeyPair(argparse.Action):
    """Created this argparse action to save kwargs
    to a dict, to be passed to pandas.read_csv()
    this functionality will be developed in the future.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
