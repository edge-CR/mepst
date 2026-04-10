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

from scipy.special import softmax
from tqdm import tqdm
from boolean.boolean import (
    _TRUE as _TRUE_CLASS,
    _FALSE as _FALSE_CLASS,
)

from . import views
from . import perturbations
from .cana import cana_values_to_sign
from .utils import (
    size_of_powerset,
    powerset,
    join_powersets,
    project,
)
from .evaluation import check_single_target_reachable_from_source
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
        title="Bounds",
        description="Ways of limiting the space of edgetic perturbation candidates.",
    )
    bounds_group.add_argument(
        "-rpt",
        "--reverse_powerset_traversal",
        action="store_true",
        help="""Start performing all perturbations and then start exploring subsets.""",
    )
    bounds_group.add_argument(
        "-apsb",
        "--auto-powerset-size-bound",
        action="store_true",
        help="""If passed, then `powerset_size_bound = min(len(fes)-1, powerset_size_bound)`.""",
    )
    bounds_group.add_argument(
        "-psb",
        "--powerset-size-bound",
        type=int,
        default=None,
        help="""Maximum/minimum cardinal to be considered when exploring subsets of the powerset.""",
    )
    bounds_group.add_argument(
        "-mcps",
        "--max-considered-perturbation-sets",
        type=int,
        default=None,
        help="""Number of candidates to consider for the powerset. 
        WARNING: The powerset of k items has 2^k - 1 elements !""",
    )
    bounds_group.add_argument(
        "-exhexpl",
        "--exhaustive-exploration",
        action="store_true",
        help="""If specified, all perturbations (within the given bounds) are tested.
        Otherwise, stop at the first successful perturbation.""",
    )
    bounds_group.add_argument(
        "-ocef",
        "--only-check-exhaustive-fes",
        action="store_true",
        help=""" After computing a certified minimal FES, compute all possible minimal FES.
        Subsequently check if all of them work. Note that this argument takes precedence over 
        others such as `--powerset-size-bound` and `--reverse_powerset-traversel`.""",
    )
    parser.add_argument(
        "-cl",
        "--cycle-limit",
        type=int,
        default=10_000_000,
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
        # nargs="*",
        # default=["targeted"],
        default="targeted",
        help="""`cutting` aims to remove the interaction, `flipping` negates the context, `targeted` monotonically favours the target.""",
    )
    # parser.add_argument(
    #    "-heu",
    #    "--heuristic",
    #    required=True,
    #    choices=tuple(_HEURISTIC_PRIORITIES),
    #    help="""Maximum number of iterations to perform when approximating the minimum Feedback Edge Set.""",
    # )
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
        deltas = (source.fillna("*") != target.fillna("*")).pipe(
            lambda s: s.index[s].sort_values().to_list()
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
        fes_opt = fes_iter[-1]
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
        psb = None
        if args.auto_powerset_size_bound:
            if args.powerset_size_bound:
                psb = min(len(fes) - 1, args.powerset_size_bound)
            else:
                psb = len(fes) - 1
        # assert psb <= len(fes) - 1
        FES = list(enum_fes_view)
        if args.only_check_exhaustive_fes:

            def make_search_space():
                return FES

        else:

            def make_search_space():
                possible_powersets = [
                    powerset(
                        _fes,
                        subset_size_bound=psb,
                        reverse=args.reverse_powerset_traversal,
                    )
                    for _fes in FES
                ]
                return join_powersets(possible_powersets)

        ### END CYCLES

        for i in range(2):
            if i:  # Flip source and target
                ia, ib = ib, ia
                source = iattrs.loc[ia, :]
                target = iattrs.loc[ib, :]
            # print(ia, ib)

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
            src_cstr = source.replace("*", np.nan).dropna().astype(int).to_dict()
            tgt_cstr = target.replace("*", np.nan).dropna().astype(int).to_dict()

            _p_bound = itertools.count()
            if _mcps := args.max_considered_perturbation_sets:
                _p_bound = range(_mcps)
            pbar.update(1)
            search_space = make_search_space()
            for candidate_n, (n_strat, _prior_interventions) in zip(
                _p_bound, enumerate(search_space)
            ):
                success = False
                perturb_func = _perturbation_types[args.perturbation_type]
                match args.perturbation_type:
                    case "cutting":
                        mutant, _posterior_interventions = perturb_func(
                            bn, _prior_interventions
                        )
                    case "flipping":
                        mutant, _posterior_interventions = perturb_func(
                            bn,
                            _prior_interventions,
                            a,
                            args.star_perturb,
                        )
                    case "targeted":
                        mutant, _posterior_interventions = perturb_func(
                            bn, _prior_interventions, cana_map, b
                        )
                    case "strict":
                        mutant, _posterior_interventions = perturb_func(
                            bn, _prior_interventions, b
                        )

                ## NEW VERSION : early stopping: if we have more than one attractor, then we have failed
                success = check_single_target_reachable_from_source(
                    mutant, common, target=b
                )
                if success:
                    with open(
                        args.out_dir / f"{_bnf_stem}{_output_suffixes[1]}", "a"
                    ) as f:
                        _outdata = {
                            "source_id": ia,
                            "target_id": ib,
                            "candidate_n": candidate_n,
                            "success": success,
                            "n_control_cycles": len(pos_control_cycles),
                            "fes": fes_opt,
                            "certified_minimal_fes": (
                                fes_view.cur_model.optimality_proven
                                if hasattr(fes_view, "cur_model")
                                else None
                            ),
                            "n_equivalent_fes": len(FES),
                            "canalisations_in_delta": delta_cana,
                            "source": a,
                            "target": b,
                            "perturbation_type": args.perturbation_type,
                            "prior_perturbations": _prior_interventions,
                            "posterior_perturbations": _posterior_interventions,
                        }
                        jl = json.dumps(_outdata, cls=GenericEncoder)
                        f.write(f"{jl}\n")
                    if not args.exhaustive_exploration:
                        break


#

if __name__ == "__main__":
    main()
