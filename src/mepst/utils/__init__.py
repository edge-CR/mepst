import math
import itertools

import pandas as pd
import networkx as nx

from .._types import _PathLike, _OptionalDict
from . import bonesis_helpers, customobjs, formatting, managers, math
from .customobjs import Path, Timer, first_arg_is_path, dont_overwite_if_exists


def size_of_powerset(iterable, subset_size_bound=None, reverse=False):
    """count elements in the powerset"""
    s = list(iterable)
    ms = len(s)
    R = range(1, min(ms, subset_size_bound or ms) + 1)
    if reverse:
        R = reversed(range(max(1, subset_size_bound or 1), ms + 1))
    return sum(math.comb(ms, r) for r in R)


def powerset(iterable, subset_size_bound=None, reverse=False):
    """ """
    s = list(iterable)
    ms = len(s)
    R = range(1, min(ms, subset_size_bound or ms) + 1)
    if reverse:
        R = reversed(range(max(1, subset_size_bound or 1), ms + 1))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in R)


def join_powersets(powersets):
    """ """
    visited = set()
    for parallel in zip(*powersets):
        for el in parallel:
            if el not in visited:
                visited.add(el)
                yield el


def project(bnx, nodes, safe=False):
    """type(bnx) := BooleanNetwork | dict"""
    cls = bnx.__class__
    domain = nodes
    if safe:
        domain = set(bnx).intersection(nodes)
    return cls({n: bnx[n] for n in domain})


def cast_trap_space_to_marker(trap_space):
    marker = trap_space.copy()
    for k, v in trap_space.items():
        if isinstance(v, str):
            marker.pop(k)
    return marker


def cast_trap_space_to_int(trap_space):
    marker = trap_space.copy()
    for k, v in trap_space.items():
        if isinstance(v, str):
            marker[k] = -1
    return marker


def cycles_to_edgelist(cycles, create_using=nx.MultiDiGraph):
    edges = []
    lookup = set()
    for cycle in cycles:
        for s, t, r in cycle:
            if (s, t, r) not in lookup:
                lookup.add((s, t, r))
                edges.append((s, t, dict(sign=r)))
    return edges


@first_arg_is_path
def parse_data_directory(
    data_dir: _PathLike,
    glob_pattern: str = "*.?sv",
    _globals: _OptionalDict = None,
    **csv_kwargs,
):
    """Parse a whole directory of csv/tsv files to pandas data frames
    by default, index column is supposed to be the fist (index_col=0).
    This can be overridden using **csv_kw"""
    if "index_col" not in csv_kwargs:
        csv_kwargs["index_col"] = 0
    data_frames = Bunch(
        **{
            file.name.replace(file.suffix, "").replace(" ", "_"): pd.read_csv(
                file.resolve(),
                sep=(
                    "\t" if "t" in file.suffix.lower() else ","
                ),  # This adaptation using the file name might be fragile
                **csv_kwargs,
            )
            for file in data_dir.glob(glob_pattern)
        }
    )

    if _globals:
        for frame in data_frames:
            _globals[frame] = data_frames[frame]

    return data_frames
