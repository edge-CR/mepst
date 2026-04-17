import math
import itertools

import pandas as pd
import networkx as nx

# from . import bonesis_helpers, customobjs, formatting
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
