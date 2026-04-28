#!/usr/bin/env python
# coding: utf-8
""" """

import sys
import math
import argparse
import datetime
import functools
import itertools
import json
import random
import collections
from pathlib import Path

import mpbn
import boolean
import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.special import softmax
from scipy.constants import golden_ratio
from tqdm import tqdm
from boolean.boolean import (
    _TRUE as _TRUE_CLASS,
    _FALSE as _FALSE_CLASS,
)

from . import views
from . import perturbations

# from . import sa
from .cana import cana_values_to_sign
from .evaluation import check_single_target_reachable_from_source
from .utils import (
    size_of_powerset,
    powerset,
    project,
    cast_trap_space_to_int,
)
from .utils.formatting import GenericEncoder

_TRUE = _TRUE_CLASS()
_FALSE = _FALSE_CLASS()
# _HEURISTIC_PRIORITIES = {
#    "frac": [
#        "fraction_of_cycles",
#        "cana_diff",
#    ],
#    "cana": [
#        "cana_diff",
#        "fraction_of_cycles",
#    ],
# }
_perturbation_types = {
    "cutting": perturbations.cutting_edgetic_perturbations,
    "flipping": perturbations.flipping_edgetic_perturbations,
    "targeted": perturbations.targeted_edgetic_perturbations,
    "strict": perturbations.strict_targeted_edgetic_perturbations,
}
_output_suffixes = (
    ".config",
    ".out",
    "cycles.asp",
    "fes.asp",
)
_exports = (
    ("config", _output_suffixes[0]),
    ("output", _output_suffixes[1]),
)
_extras = (
    ("cycles", _output_suffixes[2]),
    ("fes", _output_suffixes[3]),
)
_possible_extras = tuple(k for k, v in _extras)
_perturb_star = (
    ("t", lambda: _TRUE),
    ("f", lambda: _FALSE),
    ("r", lambda: random.choice([_FALSE, _TRUE])),
)
_perturb_star_opts = tuple(k for k, v in _perturb_star)
_perturb_star_map = dict(_perturb_star)


def exponential_chooling_scheme(alpha, T, k):
    """typically alpha ~ [0.7, 0.9]"""
    return np.pow(alpha, k) * T


def energy_mismatched_markers(
    bn, source, target, max_attrs=None, norm_delta=False, norm_card=False
):
    ndelta = sum(1 for n in bn if source[n] != target[n])
    global_energy = 0
    Ae = enumerate(bn.attractors(reachable_from=source, star=-1))
    if max_attrs is not None:
        for _, (i, a) in zip(range(max_attrs), Ae):
            local_energy = sum(1 for n in bn if a[n] != target[n])
            if norm_delta:
                local_energy /= ndelta
            global_energy += local_energy
    else:
        for i, a in Ae:
            local_energy = sum(1 for n in bn if a[n] != target[n])
            if norm_delta:
                local_energy /= ndelta
            global_energy += local_energy
    if norm_card:
        global_energy /= i + 1
    return global_energy


def sample_valid_subset(
    prior_sets: list[tuple[str, str, int]],
    weights: pd.Series | None = None,
    set_probs: pd.Series | None = None,
    expected_cardinal: int | None = None,
    rng: np.random.Generator | int | None = None,
):
    """ """
    rng = np.random.default_rng(rng)
    m = len(prior_sets)
    if set_probs is None:
        set_probs = np.ones(m) / m
    else:
        set_probs = np.asarray(set_probs, dtype=float)
        set_probs = set_probs / set_probs.sum()

    i = rng.choice(m, p=set_probs)
    S = prior_sets[i]
    N = len(S)
    nd = N - 1
    if expected_cardinal:
        nd = min(expected_cardinal, nd)

    if weights is None:
        weights = pd.Series(np.ones(N), index=S, name="weights") / N
    adj_weights = weights.copy()
    adj_weights = adj_weights.clip(lower=0).fillna(0.0)
    w = adj_weights.loc[pd.MultiIndex.from_tuples(S)]

    if w.sum() == 0:
        return []

    lam = np.log(np.exp(w).sum() / (N - nd))
    p = 1.0 - np.exp(-lam * w)
    beta = nd / p.sum()
    p *= beta
    p = p.clip(lower=0, upper=1)
    mask = rng.random(len(p)) < p
    return i, w.index[mask].sort_values().to_list()


def sample_valid_subset2(
    prior_sets: list[tuple[str, str, int]],
    weights: pd.Series | None = None,
    set_probs: pd.Series | None = None,
    expected_cardinal: int | None = None,
    rng: np.random.Generator | int | None = None,
):
    """ """
    rng = np.random.default_rng(rng)
    m = len(prior_sets)
    if set_probs is None:
        set_probs = np.ones(m) / m
    else:
        set_probs = np.asarray(set_probs, dtype=float)
        set_probs /= set_probs.sum()

    i = rng.choice(m, p=set_probs)
    S = prior_sets[i]
    N = len(S)
    nd = N - 1
    if expected_cardinal:
        nd = min(expected_cardinal, nd)

    if weights is None:
        weights = pd.Series(np.ones(N), index=S, name="weights") / N
    adj_weights = weights.copy()
    adj_weights = adj_weights.clip(lower=0).fillna(0.0)
    w = adj_weights.loc[pd.MultiIndex.from_tuples(S)]
    w = w / w.sum()

    if w.sum() == 0:
        return []

    # lam = np.log(np.exp(w).sum() / (N - nd))
    # p = 1.0 - np.exp(-lam*w)
    p = w
    beta = nd / p.sum()
    p *= beta
    p.clip(lower=0, upper=1)
    mask = rng.random(len(p)) < p
    return i, w.index[mask].sort_values().to_list()


def sample_valid_subset_surely(
    prior_sets: list[tuple[str, str, int]],
    weights: pd.Series | None = None,
    set_probs: pd.Series | None = None,
    expected_cardinal: int | None = None,
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 1000,
):
    sam = []
    attempts = 0
    while not sam and attempts < max_attempts:
        attempts += 1
        i, sam = sample_valid_subset(
            prior_sets=prior_sets,
            weights=weights,
            set_probs=set_probs,
            expected_cardinal=expected_cardinal,
            rng=rng,
        )
    return i, sam


def main():
    """ """
    parser = argparse.ArgumentParser(
        description="""sttepc : Source-Target Termporary Edgetic Perturbation Control""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group(title="IO", description="Input/Output Options")
    io_group.add_argument(
        "-bn",
        "--boolean-network",
        required=True,
        type=lambda x: Path(x).resolve(),
        help="""A `.bnet` file with the boolean network to analyse""",
    )
    io_group.add_argument(
        "-ig",
        "--influence-graph",
        type=lambda x: Path(x).resolve(),
        help="""The influence graph of the given BN (used to avoid computing it).""",
    )
    io_group.add_argument(
        "--out-dir",
        type=lambda x: Path(x).resolve(),
        help="""(Optional) path to a directory to store the results. 
        If left unspecified, the `-bn`'s parent directory will be used.""",
    )
    io_group.add_argument(
        "-p",
        "--prefix",
        type=str,
        help=""" Prefix to distinguish experiments.""",
    )
    io_group.add_argument(
        "-ee",
        "--extra-exports",
        nargs="+",
        choices=_possible_extras,
        default=tuple(),
        help="""Extra exports (on top of )""",
    )
    io_group.add_argument(
        "-eo",
        "--export-only",
        action="store_true",
        help="""Perform no calculations, export all the possible extra ASP files, run no further calculations.""",
    )
    io_group.add_argument(
        "-infc",
        "--include-non-fes-cana",
        action="store_true",
        help="""Include canalisations that are differentially """,
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=None,
        help="""Number of threads to be used by clingo.""",
    )
    bounds_group = parser.add_argument_group(
        title="Simulated Annealing",
        description="Parameters for the simulated annealing part.",
    )
    bounds_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="""Seed for the pseudo random number generator (numpy.random.default_rng(seed)).""",
    )
    bounds_group.add_argument(
        "-gs",
        "--grid-size",
        type=int,
        default=3000,
        help="""(Maximum) number of simulated annealing steps to perform.""",
    )
    bounds_group.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.995,
        help=""" Damping factor for the exponential cooling.""",
    )
    # bounds_group.add_argument(
    #    "-mf1",
    #    "--momentum-factor-1",
    #    type=float,
    #    default=None,
    #    help=""" Multiplier for the weight of the current FES, before normalisation. """,
    # )
    bounds_group.add_argument(
        "-emf",
        "--edge-momentum-factor",
        type=float,
        default=None,
        help=""" Multiplier for the weights of the current edges, before normalisation. """,
    )
    bounds_group.add_argument(
        "-b",
        "--backward",
        action="store_true",
        help=""" Backward exploration """,
    )
    # bounds_group.add_argument(
    #    "--alpha",
    #    type=float,
    #    default=0.997,
    #    help=""" Damping factor for the exponential cooling.""",
    # )
    bounds_group.add_argument(
        "-aew",
        "--adaptive-edge-weights",
        action="store_true",
        help="""Update edge weights during the annealing simulation, according to the edge improvement factor.""",
    )
    bounds_group.add_argument(
        "-eif",
        "--edge-improvement-factor",
        type=float,
        default=1.0005,
        help="""Only used whenever `--adaptive-edge-weights` is passed.
        1.0 has no effect, 1.1 means increasing the probability by 10 percent (before rescaling).""",
    )
    parser.add_argument(
        "-cl",
        "--cycle-limit",
        type=int,
        default=1000,
        help="""Maximum number of positive cycles to enumerate.""",
    )
    parser.add_argument(
        "-ct",
        "--cycle-type",
        choices=views.CyclesView._inclusions,
        default="subset",
        help="""How to consider deltas""",
    )
    parser.add_argument(
        "-ma",
        "--max-attractors",
        type=int,
        default=None,
        help="""Maximum number of attractors to consider, note that this will translate into a quadratic number of comparisons.""",
    )
    parser.add_argument(
        "-sa",
        "--sample-attractors",
        type=int,
        default=None,
        help="""Number of attractors to sample. If specified, must be lesser than `-ma`""",
    )
    parser.add_argument(
        "-to",
        "--timeout",
        type=int,
        default=30,
        help="""Maximum number of seconds between successive clingo solutions.
        Passed to `views.CyclesView` and `views.MinimalFeedbackArcSetView`.
        Note that this time is an upper bound on a computation that will be repeated ~ max_attractors^2 times.""",
    )
    parser.add_argument(
        "-mfes",
        "--max-fes-iter",
        type=int,
        default=100,
        help="""Maximum number of iterations to perform when approximating the minimum Feedback Edge Set.""",
    )
    parser.add_argument(
        "-mefes",
        "--max-enum-fes",
        type=int,
        default=100,
        help="""Maximum number of iterations to perform when approximating the minimum Feedback Edge Set.""",
    )
    parser.add_argument(
        "-pt",
        "--perturbation-type",
        choices=tuple(_perturbation_types),
        default="targeted",
        help="""`cutting` aims to remove the interaction, `flipping` negates the context, `targeted` and `strict` monotonically favour the target.
        `targeted` negates canalisations whenever the node in the target attractor is free (MTS), whereas strict only perturbs edges when the node in the target attractor is fixed.""",
    )
    parser.add_argument(
        "-sp",
        "--star-perturb",
        choices=_perturb_star_opts,
        default=_perturb_star_opts[0],
        help="""How should free variables be perturbed? (For `flipping` perturbations only).""",
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        action="store_true",
        help="""Enable debug mode.""",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="""Run without checking if the output files exist.""",
    )

    _outd = {}
    args = parser.parse_args(sys.argv[1:])
    rng = np.random.default_rng(args.seed)
    _parsed_args = vars(args)
    _outd["parsed_args"] = _parsed_args
    ### START ARG PROCESSING
    if args.export_only or args.debug:
        args.extra_exports = _possible_extras
    args.exports = _exports + tuple(
        (k, v) for k, v in _extras if k in args.extra_exports
    )
    export_dict = dict(args.exports)
    # sys.exit(f"{export_dict=}")
    _bnf = args.boolean_network
    _bnf_stem = _bnf.name.replace(_bnf.suffix, "")
    if args.prefix:
        _bnf_stem = f"{args.prefix}__{_bnf_stem}"

    if args.out_dir is None:
        args.out_dir = _bnf.parent
    if not args.force:
        for _name, _compl in args.exports:
            if _name in dict(_extras).keys():
                if _outfile := next(args.out_dir.glob(f"{_bnf_stem}*{_compl}"), None):
                    sys.exit(f"There is at least one extra file {_outfile=}, aborting")
            elif (_outfile := args.out_dir / f"{_bnf_stem}{_compl}").exists():
                sys.exit(f"Will not overwrite existing result {_outfile=}")
    # if (_outfile := args.out_dir / f"{_bnf_stem}.fail").exists():
    #    sys.exit(f"Will not overwrite existing result {_outfile=}")
    # if (_outfile := args.out_dir / f"{_bnf_stem}_cycles.asp").exists():
    #    sys.exit(f"Will not overwrite existing result {_outfile=}")
    if args.sample_attractors:
        assert args.sample_attractors < args.max_attractors
    ### END ARG PROCESSING
    _processed_args = vars(args)
    _outd["processed_args"] = _processed_args

    bn = mpbn.load(args.boolean_network.as_posix()).simplify()

    if args.influence_graph:
        _ig = args.influence_graph
        # assert _bnf_stem == _ig.name.replace(_ig.suffix, '')
        ig = nx.nx_agraph.read_dot(args.influence_graph.as_posix())
        bn_in_ig = all(n in ig for n in bn) and len(bn) > 0
        ig_in_bn = all(n in bn for n in ig) and len(ig) > 0
        assert bn_in_ig and ig_in_bn, "BN and IG mismatch !"
    else:
        ig = bn.influence_graph()

    _bound = itertools.count()
    if args.max_attractors:
        _bound = range(args.max_attractors)
    _iattrs = []
    for _an, A in tqdm(
        zip(_bound, bn.attractors(star=np.nan)),
        desc="Computing attractors",
    ):
        _iattrs.append(A)
    iattrs = pd.DataFrame(_iattrs)
    if _sna := args.sample_attractors:
        iattrs = iattrs.sample(_sna)
    nA = iattrs.index.shape[0]
    canalisations = next(iter(views.CanalisationsView.from_bn(bn, arity=4)))

    _outd["n_attractors"] = nA
    _outd["canalisations"] = canalisations
    with open(args.out_dir / f"{_bnf_stem}{_output_suffixes[0]}", "a") as f:
        f.write(json.dumps(_outd, cls=GenericEncoder))

    ncomparisons = nA * nA - nA
    pbar = tqdm(total=(nA * nA - nA))
    for ia, ib in itertools.combinations(iattrs.index, 2):
        source = iattrs.loc[ia, :]
        target = iattrs.loc[ib, :]
        #### BEGIN OPT : (do not repeat the computations common to (s -> t) and (t -> s)
        deltas = (
            (source.fillna("*") != target.fillna("*"))
            .pipe(
                lambda s: (
                    s
                    if not args.minimal_delta
                    else (s & ~(source.isna() | target.isna()))
                )
            )
            .pipe(lambda s: s.index[s].sort_values().to_list())
        )
        if args.debug:
            print(f"{len(deltas)=}", flush=True)

        delta_graph = nx.subgraph(ig, deltas)
        delta_cana = [
            cana for cana in canalisations if all(c in delta_graph for c in cana[:2])
        ]
        differential_cana_arcs = [cana_values_to_sign(c4) for c4 in delta_cana]
        cana_map = {(dc[0], dc[1]): (dc[2], dc[3]) for dc in delta_cana}

        if args.debug:
            print(f"{nx.is_directed_acyclic_graph(delta_graph)=}", flush=True)
            nx.nx_agraph.write_dot(
                delta_graph,
                (args.out_dir / f"{_bnf_stem}_{ia}-{ib}_delta_graph.dot").as_posix(),
            )
        ### START CYCLES
        _cycles_kwargs = dict(
            node_subset=deltas,
            inclusion=args.cycle_type,
            sign="+",
            limit=args.cycle_limit,
            quiet=True,
            timeout=args.timeout,
            fail_if_timeout=False,
        )
        if args.threads is not None:
            _cycles_kwargs["parallel"] = args.threads

        pos_control_cycles = []
        cycles_view = views.CyclesView.from_nx_influence_graph(
            delta_graph, **_cycles_kwargs
        )
        if (_extra := "cycles") in export_dict:
            cycles_view.configure(ground=False)
            cycles_view.control.export_rules(
                (
                    args.out_dir / f"{_bnf_stem}_{ia}-{ib}_{export_dict[_extra]}"
                ).as_posix()
            )
        if not args.export_only:
            if args.debug:
                print("Enumerating cycles...", end="\t", flush=True)
            for pos_cycle in cycles_view:
                pos_control_cycles.append(pos_cycle)
            if args.debug:
                print(f"Done. Found {len(pos_control_cycles)=}", flush=True)

        _fes_kwargs = _cycles_kwargs.copy()
        for k in ["node_subset", "sign"]:
            _ = _fes_kwargs.pop(k)

        # mfes = []
        fes_iter = []

        def fes_callback(model):
            fes_iter.append(model)

        _fes_kwargs.update(
            dict(
                mode="optN",
                intermediate_model_cb=fes_callback,
            )
        )
        if maxnfes := args.max_fes_iter:
            _fes_kwargs["limit"] = maxnfes

        fes_view = views.FeedbackArcSetView.from_cycles(
            pos_control_cycles, **_fes_kwargs
        )
        # print(_fes_kwargs)
        # with open(args.out_dir / f"{_bnf_stem}_fes.asp", "w") as f:
        #    print(fes_view.standalone(), file=f)
        if (_extra := "fes") in export_dict:
            fes_view.configure(ground=False)
            fes_view.control.export_rules(
                (
                    args.out_dir / f"{_bnf_stem}_{ia}-{ib}_{export_dict[_extra]}"
                ).as_posix()
            )
        if not args.export_only:
            for fes in fes_view:
                break
        if args.export_only:
            continue
        # Should `fes_opt` be our best guess or our first guess (more options vs better estimation) ?
        # Shouldn't we have a fallback ?

        ### START PRIOR SET CONSTRUCTION
        fes_opt = fes_iter[-1]
        # if args.only_check_exhaustive_fes:
        _efes_kwargs = _fes_kwargs.copy()
        for k in ["limit", "mode"]:
            _ = _efes_kwargs.pop(k)
        _efes_kwargs["mode"] = "solve"
        _efes_kwargs["limit"] = args.max_enum_fes
        enum_fes_view = views.FeedbackArcSetView.from_cycles(
            pos_control_cycles, **_efes_kwargs
        )
        enum_fes_view.custom(
            f":- #count {{ S,T,R : remove(S,T,R)}} = SOLSIZE, SOLSIZE > {len(fes)}."
        )
        search_space = list(enum_fes_view)
        search_space_size = len(search_space)
        perturb_type = args.perturbation_type
        perturb_func = _perturbation_types[perturb_type]
        valid_prior_sets = search_space
        _edges = sorted(set(itertools.chain.from_iterable(valid_prior_sets)))
        n_edges = len(_edges)
        n_prior_sets = len(valid_prior_sets)

        base_set_probs = np.ones(len(valid_prior_sets)) / len(valid_prior_sets)
        base_weight = pd.Series(
            np.ones(n_edges) / n_edges, index=_edges, name="uniform"
        )
        omega = base_weight.sort_index().index
        _set_indexer = range(len(valid_prior_sets))
        fes_indicator = pd.DataFrame(
            [{k: k in s for k in omega} for s in valid_prior_sets], columns=omega
        )
        similarity = 1 - pd.DataFrame(
            distance.cdist(fes_indicator, fes_indicator, metric="jaccard"),
            index=_set_indexer,
            columns=_set_indexer,
        )
        ### Make the distance matrix communicating (repair)
        G_fes = nx.from_pandas_adjacency(similarity, create_using=nx.DiGraph)
        is_sc = nx.is_strongly_connected(G_fes)
        if not is_sc:
            prior_nonzero_min = similarity[similarity > 0.0].min(axis=None).item()
            similarity[np.isclose(similarity, 0.0)] = prior_nonzero_min
            G_fes = nx.from_pandas_adjacency(similarity, create_using=nx.DiGraph)
            is_sc = nx.is_strongly_connected(G_fes)
            assert is_sc
        ### End distance matrix reparation

        nosim = similarity.apply(lambda col: col / col.sum())
        ### END PRIOR SET CONSTRUCTION

        for sst in range(2):
            if sst:  # Swap Source and Target
                ia, ib = ib, ia
                source = iattrs.loc[ia, :]
                target = iattrs.loc[ib, :]

            common = source.dropna().to_dict()
            a = (
                source.fillna("*")
                .map(lambda el: el if isinstance(el, str) else int(el))
                .to_dict()
            )
            b = (
                target.fillna("*")
                .map(lambda el: el if isinstance(el, str) else int(el))
                .to_dict()
            )
            ca = cast_trap_space_to_int(a)
            cb = cast_trap_space_to_int(b)
            src_cstr = source.replace("*", np.nan).dropna().astype(int).to_dict()
            tgt_cstr = target.replace("*", np.nan).dropna().astype(int).to_dict()

            ### START SIMULATED ANNEALING
            To = T = (len(deltas) + len(valid_prior_sets[0])) * 100

            initial_fes = valid_prior_sets[0]
            # this replaces scores.index in the notebook representation.
            sa_tested_perturbations = []
            sa_sampled_energies = []
            rsa_tested_perturbations = []
            rsa_sampled_energies = []
            visited_perturbations_witness = set()
            fes_card = len(fes)
            current_solution = initial_fes
            global_best_solution = fes
            global_best_energy = To
            global_best_posterior = None
            current_energy = To
            current_satE = To
            total = GS = args.grid_size
            # cs = functools.partial(sa.exponential_chooling_scheme, args.alpha, GS)
            sampled_set_index = None
            sm = None
            weight_history = [base_weight.rename(0)]
            p_accept_bad = []
            n_excursions = 0

            # i = 0
            solution_found = False
            for i in range(1, total + 1):
                if args.backward:
                    _ssf = (total - i) / total
                else:
                    _ssf = i / total
                size = int(_ssf * fes_card)
                if 0 == size:
                    size = 1

                current_set_probs = base_set_probs.copy()
                current_edge_weights = base_weight.copy()
                if sampled_set_index is not None:
                    # current_set_probs[sampled_set_index] *= args.momentum_factor_1
                    current_set_probs = nosim[sampled_set_index].copy()
                    current_set_probs /= current_set_probs.sum()
                sampled_set_index = rng.choice(n_prior_sets, p=current_set_probs)
                _current_set = valid_prior_sets[sampled_set_index]
                if (sm is not None) and (args.edge_momentum_factor):
                    current_edge_weights[
                        pd.MultiIndex.from_tuples(sm)
                    ] *= args.edge_momentum_factor
                current_edge_weights = current_edge_weights.loc[
                    pd.MultiIndex.from_tuples(_current_set)
                ]
                current_edge_weights = current_edge_weights / current_edge_weights.sum()
                sm = rng.choice(
                    current_edge_weights.index,
                    p=current_edge_weights.values,
                    replace=False,
                    size=size,
                )

                smt = tuple(sm)
                # visited_perturbations_witness.add(smt)
                match perturb_type:
                    case "cutting":
                        mutant, _posterior_interventions = perturb_func(bn, sm)
                    case "flipping":
                        mutant, _posterior_interventions = perturb_func(
                            bn,
                            sm,
                            a,
                            args.star_perturb,
                        )
                    case "targeted":
                        mutant, _posterior_interventions = perturb_func(
                            bn, sm, cana_map, b
                        )
                    case "strict":
                        mutant, _posterior_interventions = perturb_func(bn, sm, b)

                satE = energy_mismatched_markers(
                    mutant, ca, cb, norm_card=True, norm_delta=True
                )
                nsm = len(sm)
                carE = len(sm)  # / fes_card
                totalE = satE + carE
                # rsa_sampled_energies.append((satE, carE))
                # rsa_tested_perturbations.append(sm)
                # T = cs(i)
                T *= args.alpha
                # print(f"card={len(sm)},{satE=}", end=",")
                # if totalE <= current_energy and satE <= current_satE:
                if satE <= current_satE:
                    # print("better", end=",")
                    # sa_sampled_energies.append((satE, carE))
                    # sa_tested_perturbations.append(sm)
                    current_solution = sm
                    current_energy = totalE
                    current_satE = satE
                    # strict sat enables dodging numerical errors
                    strict_sat = check_single_target_reachable_from_source(
                        mutant, common, target=b
                    )
                    if np.isclose(satE, 0) and strict_sat:
                        # print("SAT", end=",")
                        if args.adaptive_edge_weights:
                            base_weight[
                                pd.MultiIndex.from_tuples(sm)
                            ] *= args.edge_improvement_factor
                            base_weight /= base_weight.sum()
                            # weight_history.append(base_weight.rename(i))
                        if not args.backward:
                            with open(
                                args.out_dir / f"{_bnf_stem}{_output_suffixes[1]}", "a"
                            ) as f:
                                _outdata = {
                                    "heuristic_scheme": "forward_pseudo_simulated_annealing",
                                    "source_id": ia,
                                    "target_id": ib,
                                    "candidate_n": i,
                                    "source": a,
                                    "target": b,
                                    "perturbation_type": perturb_type,
                                    "prior_perturbations": sm,
                                    "posterior_perturbations": _posterior_interventions,
                                    "fes_size": len(fes),
                                    "certified_minimal_fes": (
                                        fes_view.cur_model.optimality_proven
                                        if hasattr(fes_view, "cur_model")
                                        else None
                                    ),
                                    "n_deltas": len(deltas),
                                }
                                jl = json.dumps(_outdata, cls=GenericEncoder)
                                f.write(f"{jl}\n")
                            pbar.update(1)
                            break
                        # elif current_energy < global_best_energy:
                        elif current_energy <= global_best_energy:
                            global_best_solution = current_solution
                            global_best_energy = current_energy
                            global_best_posterior = _posterior_interventions
                            solution_found = True
                else:
                    _p_accept_bad = np.clip(
                        np.exp(-(totalE - current_energy) / T), a_min=0, a_max=1
                    )
                    # p_accept_bad.append((i, T, _p_accept_bad))
                    if rng.random(1).item() < _p_accept_bad:
                        current_solution = sm
                        current_energy = totalE
                        current_satE = satE
                        current_carE = carE
                        n_excursions += 1
            else:
                pbar.update(1)
                if solution_found == True:
                    with open(
                        args.out_dir / f"{_bnf_stem}{_output_suffixes[1]}", "a"
                    ) as f:
                        _outdata = {
                            "heuristic_scheme": "forward_pseudo_simulated_annealing",
                            "source_id": ia,
                            "target_id": ib,
                            "candidate_n": i,
                            "source": a,
                            "target": b,
                            "perturbation_type": perturb_type,
                            "prior_perturbations": global_best_solution,
                            "posterior_perturbations": global_best_posterior,
                            "fes_size": len(fes),
                            "certified_minimal_fes": (
                                fes_view.cur_model.optimality_proven
                                if hasattr(fes_view, "cur_model")
                                else None
                            ),
                            "n_deltas": len(deltas),
                        }
                        jl = json.dumps(_outdata, cls=GenericEncoder)
                        f.write(f"{jl}\n")
            ### END SIMULATED ANNEALING


#

if __name__ == "__main__":
    main()
