"""Helper module to develop functions related to bonesis."""

import bonesis as __bonesis
import copy
import networkx as nx


def save_to_dot(ig, path, remove_labels=True):
    """ """
    from .plotting import color_influence_graph

    ig = copy.deepcopy(ig)
    G = nx.MultiDiGraph(ig)

    for i, j, _dat in G.edges(data=True):
        if "label" in _dat:
            _dat.pop("label")

    G = color_influence_graph(G)
    nx.nx_pydot.write_dot(G, path)


def dict_from_active_and_inactive(bn, active, inactive):
    """This function creates a dictionary for simulation under
    MP semantics, given `bn` and two sets, one of "active" and
    other of "inactive" components.

    parameters:
    ----------
        * bn: An instance of any of:
                * colomoto.minibn.BooleanNetwork
                * bonesis.BoNesis
                * networkx.classes.multidigraph.MultiDiGraph
    """
    bn = bn.domain if isinstance(bn, __bonesis.BoNesis) else bn
    assert set(active).issubset(bn)
    assert set(inactive).issubset(bn)
    _active = {n: 1 for n in active}
    _inactive = {n: 0 for n in inactive}
    _determined = _active.copy()
    _determined.update(_inactive)

    return _determined


def dict_from_active_and_undetermined(bn, active, undetermined=set()):
    """This function creates a dictionary for simulation under
    MP semantics, `bn` and two sets, one of "active" and other of
    "undetermined" components.
    Those which are neither will be set to inactive (zero).

    parameters:
    ----------
        * bn: An instance of any of:
                * colomoto.minibn.BooleanNetwork
                * bonesis.BoNesis
                * networkx.classes.multidigraph.MultiDiGraph
    """
    bn = bn.domain if isinstance(bn, __bonesis.BoNesis) else bn
    assert set(active).issubset(bn)
    assert set(undetermined).issubset(bn)
    _determined = {n: int(n in active) for n in bn}
    if undetermined:
        # 'Undetermine' the necessary states :
        [_determined.pop(_) for _ in undetermined]

    return _determined


def configuration_from_dict(bo: __bonesis.BoNesis, conf_dict, name=None):
    """Take a bonesis.BoNesis object and a dict,
    create a configuration and return it"""
    if not isinstance(bo, __bonesis.BoNesis):
        raise TypeError(f"Expected a BoNesis object, got {type(bo)} instead.")

    assert set(conf_dict).issubset(bo.domain)
    _obs = bo.cfg(name=name) if name is not None else bo.cfg()
    if not set(int(val) for val in conf_dict.values()).issubset({0, 1}):
        raise ValueError("`conf_dict` contains values other than {0, 1}")
    for node in conf_dict:
        _obs[node] = conf_dict[node]

    return _obs


def configuration_from_active_and_undetermined(
    bo: __bonesis.BoNesis, active, undetermined=set(), name=None
):
    """Chain the call to `dict_from_active_and_undetermined`
    and `configuration_from_dict`"""
    _conf_dict = dict_from_active_and_undetermined(
        bo, active, undetermined=undetermined
    )
    return configuration_from_dict(bo, _conf_dict, name=name)
