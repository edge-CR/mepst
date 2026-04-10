import random

import boolean

from boolean.boolean import (
    _TRUE as _TRUE_CLASS,
    _FALSE as _FALSE_CLASS,
)

_TRUE = _TRUE_CLASS()
_FALSE = _FALSE_CLASS()


_perturb_star = (
    ("t", lambda: _TRUE),
    ("f", lambda: _FALSE),
    ("r", lambda: random.choice([_FALSE, _TRUE])),
)
_perturb_star_opts = tuple(k for k, v in _perturb_star)
_perturb_star_map = dict(_perturb_star)


def no_perturbation(bn, *args, **kwargs):
    """no perturbation, used for API consistency."""
    return bn


def cutting_edgetic_perturbations(bn, edges, negative_edge=-1):
    """If the node has a single cause (a Literal), then no modification takes place.
    Any of these situations may arise because of previous mutations."""
    TRUE, FALSE, NOT, AND, OR, Symbol = bn.ba.definition()
    _ALG_OPS = {AND, OR}
    mutant = bn.copy()
    real_edges = []
    for _regulator, _target, _sign in edges:
        f = mutant[_target]
        sym_to_lit = dict(zip(sorted(f.symbols), sorted(f.literals)))
        # The check is needed because previous mutations may
        if (reg_lit := sym_to_lit.get(boolean.Symbol(_regulator))) is not None:
            if f.__class__ in _ALG_OPS:
                ops = {
                    True: f.identity,
                    False: f.annihilator,
                }
                new_f = (
                    mutant[_target].subs({reg_lit: ops[reg_lit in f.args]}).simplify()
                )
                if new_f != f:
                    mutant[_target] = new_f
                    real_edges.append((_regulator, _target, _sign))

    return mutant, real_edges


def flipping_edgetic_perturbations(f, edges, context, star_perturb):
    """Return a perturbed version (shallow copy) of Boolean network `f` so that
    each one of the given `edges` are:
    - Fixed to the complement of what the given context implies (when context is constant i.e. 0 / 1)
    - To a random value (either 1 or 0) whenever the context is free ("star" in mpbn terms).

    @param f: a mpbn.MPBooleanNetwork
    @param edges: a list of tuples with (regulator, target, sign) (as encoded in BoNesis)
    @param context: a dict as returned by mpbn.MPBooleanNetwork.attractors().
    """
    # Why is ```boolean.BooleanAlgebra().definition() != bn.ba.definition()``` ?
    # i.e. why `colomoto.minibn.NOT` is different from `boolean.boolean.NOT` ?
    # because `colomoto.minibn.NOT`'s operator is `!` instead of `~`
    TRUE, FALSE, NOT, AND, OR, Symbol = f.ba.definition()
    _eval_map = {1: TRUE, 0: FALSE}
    mutant = f.copy()

    def _f_complement_eval_map(value):
        match value:
            case 1:
                return FALSE
            case 0:
                return TRUE
            case "*":
                return _perturb_star_map[star_perturb]()
            case _:
                if np.isnan(value):
                    return _perturb_star_map[star_perturb]()
                raise ValueError(f"Unknown {value=}")

    real_edges = []
    for _parent, _target, _sign in edges:
        new_f = (
            mutant[_target]
            .subs({boolean.Symbol(_parent): _f_complement_eval_map(context[_parent])})
            .simplify()
        )
        if new_f != f:
            mutant[_target] = new_f
            real_edges.append((_parent, _target, _sign))
    return mutant, real_edges


def targeted_edgetic_perturbations(bn, edges, cana_map, target):
    """Return a perturbed version (shallow copy) of Boolean network `f` so that
    each one of the given `edges` are perturbed in order to favour the reachability of target.

    @param bn: a mpbn.MPBooleanNetwork
    @param edges: a list of tuples with (regulator, target, sign) (as encoded in BoNesis)
    @param edges: a list of tuples with (regulator, target, canalising_value, canalised_value)
                  (as encoded in sttepc.views.CanalisationsView)
    @param target: a dict as returned by mpbn.MPBooleanNetwork.attractors().
    """
    TRUE, FALSE, NOT, AND, OR, Symbol = bn.ba.definition()
    _eval_map = {1: TRUE, 0: FALSE}
    mutant = bn.copy()

    real_edges = []
    for _parent, _target, _sign in edges:
        f = bn[_target]
        sym_to_lit = dict(zip(sorted(f.symbols), sorted(f.literals)))
        sign_of_sym = {s: int(isinstance(l, Symbol)) for s, l in sym_to_lit.items()}
        ps = boolean.Symbol(_parent)
        # match if the target is fixed or not
        match target[_target] in _eval_map:
            case True:
                subs_key = sign_of_sym[ps] == target[_target]
                new_f = mutant[_target].subs({ps: _eval_map[subs_key]})
            case False:
                new_f = mutant[_target]
                if (e := (_parent, _target)) in cana_map:
                    new_f = mutant[_target].subs({ps: _eval_map[not cana_map[e][0]]})
        if new_f != f:
            mutant[_target] = new_f
            real_edges.append((_parent, _target, _sign))
    return mutant, real_edges


def strict_targeted_edgetic_perturbations(bn, edges, target):
    """Return a perturbed version (shallow copy) of Boolean network `f` so that
    each one of the given `edges` are perturbed in order to favour the reachability of target.

    @param bn: a mpbn.MPBooleanNetwork
    @param edges: a list of tuples with (regulator, target, sign) (as encoded in BoNesis)
    @param target: a dict as returned by mpbn.MPBooleanNetwork.attractors().
    """
    TRUE, FALSE, NOT, AND, OR, Symbol = bn.ba.definition()
    _eval_map = {1: TRUE, 0: FALSE}
    mutant = bn.copy()

    real_edges = []
    for _parent, _target, _sign in edges:
        f = bn[_target]
        sym_to_lit = dict(zip(sorted(f.symbols), sorted(f.literals)))
        sign_of_sym = {s: int(isinstance(l, Symbol)) for s, l in sym_to_lit.items()}
        ps = boolean.Symbol(_parent)
        # match if the target is fixed or not
        match target[_target] in _eval_map:
            case True:
                subs_key = sign_of_sym[ps] == target[_target]
                new_f = mutant[_target].subs({ps: _eval_map[subs_key]})
            case False:
                new_f = mutant[_target]
        if new_f != f:
            mutant[_target] = new_f
            real_edges.append((_parent, _target, _sign))
    return mutant, real_edges
