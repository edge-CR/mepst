import time
import warnings
import tempfile

import clingo
import networkx as nx
import bonesis
from bonesis.utils import OverlayedDict, frozendict
from bonesis0.asp_encoding import py_of_symbol, symbol_of_py, symbols

from . import cana
from .utils import cycles_to_edgelist

_BONESIS_EDGE_PREDICATE: str = "in"
_DEFAULT_EXACT: str | bool = "unsigned"
_DEFAULT_INCLUSION: str | bool = "subset"


class CanalisationsView(bonesis.views.BonesisView):
    """Check the `canalising` predicate (canalising/4, canalising/3) in order to avoid unpleasant surprises !
    Here the first argument will be the regulator, to follow the convention in the interaction graphs.
    This is contrary to the convention of the ASP encoding, specially the clause/4 predicate which has the target first.
    """

    project = True
    default_enum = "auto"
    _possible_enums = ("auto", "brave", "cautious")
    __preds_ = [
        ("canalising", 4),
        ("canalising", 3),
    ]
    __asp_ = [
        "canalising(R,T,RS,1) :- clause(T,C,R,RS), #count { D: clause(T,C,D,_) } 1.",
        "canalising(R,T,RS,0) :- clause(T,_,R,RS); clause(T,C,R,RS) : clause(T,C,_,_).",
        "canalising(R,T,RS) :- canalising(R,T,RS,_).",
        # "canavals(T,R,0,1) :- canalising(T,R,-1,suf)",
    ]

    @classmethod
    def from_bn(
        cls,
        bn,
        as_values=True,
        arity=4,
        *args,
        **kwargs,
    ):
        bo = bonesis.BoNesis(bn)

        inst = cls(
            bo,
            arity=arity,
            *args,
            **kwargs,
        )
        return inst

    def __init__(self, bo, as_values=True, arity=4, *args, **kwargs):
        valid_arities = [a for p, a in self.__preds_]
        assert any(
            arity == a for a in valid_arities
        ), f"Invalid {arity=}, choose amongst {valid_arities=}"
        if arity != 4 and as_values:
            raise ValueError(f"{as_values=} is only compatible with arity=4.")
        self.arity = arity
        self.as_values = as_values
        super().__init__(bo, *args, **kwargs)
        self.bo = bo
        for constr in self.__asp_:
            self.bo.custom(constr)
        self._enum_mode = self.default_enum
        if enum := kwargs.get("enum_mode"):
            assert enum in self._possible_enums
            self._enum_mode = enum

    def configure(self, ground=True, **opts):
        args = [0]
        if (
            self.single_shot
            and hasattr(clingo, "version")
            and clingo.version() >= (5, 5, 0)
        ):
            args.append("--single-shot")
        if self.project:
            args.append("--project")
        if self.mode == "optN":
            opt_strategy = self.settings.get("clingo_opt_strategy", "usc")
            args += ["--opt-mode=optN", f"--opt-strategy={opt_strategy}"]
        elif self.mode == "solve" and self.bo.has_optimizations():
            args += ["--opt-mode=ignore"]

        settings = OverlayedDict(self.settings)
        if self.settings["solutions"] in ["subset-minimal", "subset-maximal"]:
            if parse_nb_threads(settings.get("parallel")) > 1:
                args += ["--configuration", portfolio_path("subset_portfolio")]
            args += [
                "--heuristic",
                "Domain",
                "--enum-mode",
                "domRec",
                "--dom-mod",
                "5,16" if self.settings["solutions"] == "subset-minimal" else "3,16",
            ]
        args += ["--enum-mode", self._enum_mode]

        if not self.settings["quiet"] and ground:
            print("Grounding...", end="", flush=True)
            start = time.process_time()
        self.control = self.bo.solver(*args, settings=settings, ground=False, **opts)
        self.interrupted = False
        self.configure_show()
        if ground:
            self.control.ground([("base", ())])
        if ground and not self.settings["quiet"]:
            end = time.process_time()
            print(f"done in {end-start:.1f}s")

    def configure_show(self):
        self.control.add("base", [], "#show." f"#show canalising/{self.arity}.")

    def format_model(self, model):
        atoms = model.symbols(shown=True)
        if self.as_values:
            return [cana.cana_tax_to_values(py_of_symbol(atom)) for atom in atoms]
        return [py_of_symbol(atom) for atom in atoms]


class CyclesView(bonesis.views.BonesisView):
    project = True
    _inclusions = {"cautious", "subset", "brave"}
    _signs = {"+": 1, "-": 0}
    __asp_ = [
        "{ in_cycle(X,Y,S) } :-" + f"{_BONESIS_EDGE_PREDICATE}(X,Y,S).",
        ":- in_cycle(X,Y1,_), in_cycle(X,Y2,_), Y1 != Y2.",
        ":- in_cycle(X1,Y,_), in_cycle(X2,Y,_), X1 != X2.",
        ":- not in_cycle(_,_,_).",
        ":- start(X), in_cycle(Y,_,_), X > Y.",
        "1 { start(X) : node(X), in_cycle(X,_,_) } 1.",
        "reach(X,Y) :- in_cycle(X,Y,_).",
        "reach(X,Z) :- reach(X,Y), in_cycle(Y,Z,_).",
        ":- in_cycle(X,_,_), start(S), not reach(S,X).",
    ]

    @classmethod
    def from_nx_influence_graph(
        cls,
        ig: nx.MultiDiGraph,
        sign=None,
        node_subset={},
        inclusion=_DEFAULT_INCLUSION,
        exact=_DEFAULT_EXACT,
        canonic=False,
        maxclause=4,
        *args,
        **kwargs,
    ):
        dom = bonesis.InfluenceGraph(
            ig, exact=exact, canonic=canonic, maxclause=maxclause
        )
        bo = bonesis.BoNesis(dom)

        inst = cls(
            bo,
            sign=sign,
            node_subset=node_subset,
            inclusion=inclusion,
            *args,
            **kwargs,
        )
        return inst

    def __init__(
        self,
        bo,
        sign=None,
        node_subset={},
        inclusion=_DEFAULT_INCLUSION,
        *args,
        **kwargs,
    ):
        super().__init__(bo, *args, **kwargs)
        self.project = True
        self.bo = bo
        # if "settings" in kwargs:
        #    settings = kwargs.pop()
        # for constr in self.__asp_:
        #    self.bo.custom(constr)
        constraints = []
        if sign is not None:
            assert (
                sign in self._signs
            ), f"Invalid cycle sign {sign}, available signs: {self._signs.keys()}"
            sign_constr = (
                r":- #count { X,Y,S : in_cycle(X,Y,S), S = -1 } = NNEG, NNEG \ 2 = "
                + f"{self._signs[sign]}."
            )
            constraints.append(sign_constr)
        if node_subset:
            constraints.append(
                f"in_cycle(X) :- reach(X,X), {_BONESIS_EDGE_PREDICATE}(X,Y,_)."
            )
            for node in node_subset:
                assert node in self.bo.domain, f"{node=} not in domain"
                constraints.append(f'must_include("{node}").')
            match inclusion:
                case "cautious":
                    constraints.append(":- not in_cycle(X), must_include(X).")
                case "subset":
                    constraints.append(":- in_cycle(X), not must_include(X).")
                case "brave":
                    constraints.append("includes :- in_cycle(X), must_include(X).")
                    constraints.append(":- not includes.")
                case _:
                    raise ValueError(
                        f"Unknown {inclusion=}, should be one of {self._inclusions}"
                    )
        self._constraints = constraints

    def configure(self, ground=True, **opts):
        args = [0]
        if (
            self.single_shot
            and hasattr(clingo, "version")
            and clingo.version() >= (5, 5, 0)
        ):
            args.append("--single-shot")
        if self.project:
            args.append("--project")
        if self.mode == "optN":
            opt_strategy = self.settings.get("clingo_opt_strategy", "usc")
            args += ["--opt-mode=optN", f"--opt-strategy={opt_strategy}"]
        elif self.mode == "solve" and self.bo.has_optimizations():
            args += ["--opt-mode=ignore"]

        settings = OverlayedDict(self.settings)
        if self.settings["solutions"] in ["subset-minimal", "subset-maximal"]:
            if parse_nb_threads(settings.get("parallel")) > 1:
                args += ["--configuration", portfolio_path("subset_portfolio")]
            args += [
                "--heuristic",
                "Domain",
                "--enum-mode",
                "domRec",
                "--dom-mod",
                "5,16" if self.settings["solutions"] == "subset-minimal" else "3,16",
            ]

        aspm = self.bo.aspmodel
        aspm.reset()
        facts = []
        for n in aspm.domain.nodes():
            facts.append(clingo.Function("node", symbols(n)))
        for orig, dest, data in aspm.domain.edges(data=True):
            if data["sign"] in ["ukn", "?", "0", 0]:
                args = symbols(orig, dest)
                f = "in({},{},(-1;1))".format(*args)
                facts.append(f)
            else:
                ds = data["sign"]
                if ds in ["-", "+"]:
                    ds += "1"
                s = int(ds)
                facts.append(clingo.Function("in", symbols(orig, dest, s)))
        aspm.push(facts)

        for constr in self.__asp_:
            aspm.push(aspm.encode_custom(constr))
        for constr in self._constraints:
            aspm.push(aspm.encode_custom(constr))

        if not self.settings["quiet"] and ground:
            print("Grounding...", end="", flush=True)
            start = time.process_time()
        self.control = self.aspmodel.solver(
            *args, settings=settings, ground=False, **opts
        )
        self.interrupted = False
        self.configure_show()
        if ground:
            self.control.ground([("base", ())])
        if ground and not self.settings["quiet"]:
            end = time.process_time()
            print(f"done in {end-start:.1f}s")

    def configure_show(self):
        self.control.add("base", [], "#show." "#show in_cycle/3.")

    def format_model(self, model):
        atoms = model.symbols(shown=True)
        return [py_of_symbol(atom) for atom in atoms]


class FeedbackArcSetView(bonesis.views.BonesisView):
    project = True
    default_mode = "optN"
    default_enum = "auto"
    _possible_enums = ("auto", "brave", "cautious")
    program_name = "fes"
    __asp_ = [
        "{remove(S,T,R) : ext_edge(S, T, R)}.",
        "hit(C) :- remove(S,T,R), in_cycle(C,S,T,R).",
        ":- cycle(C), not hit(C).",
    ]

    @classmethod
    def from_cycles(cls, cycles, *args, **kwargs):
        edges = cycles_to_edgelist(cycles)
        bo = bonesis.BoNesis(nx.from_edgelist(edges, create_using=nx.MultiDiGraph))
        inst = cls(bo, cycles, *args, **kwargs)
        return inst

    def __init__(self, bo, cycles, *args, **kwargs):
        if "mode" not in kwargs:
            kwargs["mode"] = self.default_mode
        super().__init__(bo, *args, **kwargs)
        self.bo = bo
        self.cycles = cycles
        self._opt = kwargs["mode"].startswith("opt")
        self._enum_mode = self.default_enum
        if enum := kwargs.get("enum_mode"):
            assert enum in self._possible_enums
            self._enum_mode = enum
        self.properties = []

    def custom(self, constraint):
        self.properties.append(constraint)

    def configure(self, ground=True, **opts):
        args = [0]
        if (
            self.single_shot
            and hasattr(clingo, "version")
            and clingo.version() >= (5, 5, 0)
        ):
            args.append("--single-shot")
        if self.project:
            args.append("--project")
        if self.mode == "optN":
            opt_strategy = self.settings.get("clingo_opt_strategy", "usc")
            args += ["--opt-mode=optN", f"--opt-strategy={opt_strategy}"]
        elif self.mode == "solve" and self.bo.has_optimizations():
            args += ["--opt-mode=ignore"]

        settings = OverlayedDict(self.settings)
        if self.settings["solutions"] in ["subset-minimal", "subset-maximal"]:
            if parse_nb_threads(settings.get("parallel")) > 1:
                args += ["--configuration", portfolio_path("subset_portfolio")]
            args += [
                "--heuristic",
                "Domain",
                "--enum-mode",
                "domRec",
                "--dom-mod",
                "5,16" if self.settings["solutions"] == "subset-minimal" else "3,16",
            ]
        args += ["--enum-mode", self._enum_mode]

        self.bo.aspmodel.reset()
        facts = []
        for i, cycle in enumerate(self.cycles):
            base_atoms = [f"ext_edge{symbol_of_py(edge)}" for edge in cycle]
            in_cycle_atoms = [f"in_cycle{(symbol_of_py((i, *edge)))}" for edge in cycle]
            in_cycle_atoms.append(f"cycle({i})")
            facts += base_atoms + in_cycle_atoms

        for constr in self.__asp_:
            self.bo.aspmodel.push(self.bo.aspmodel.encode_custom(constr))
        self.bo.aspmodel.push(facts)
        if self._opt:
            self.bo.aspmodel.push(
                self.bo.aspmodel.encode_custom("#minimize { 1,S,T,R : remove(S,T,R)}.")
            )
        for constr in self.properties:
            self.bo.aspmodel.push(self.bo.aspmodel.encode_custom(constr))

        if not self.settings["quiet"] and ground:
            print("Grounding...", end="", flush=True)
            start = time.process_time()
        self.control = self.aspmodel.solver(
            *args, settings=settings, ground=False, **opts
        )
        self.interrupted = False
        self.configure_show()
        if ground:
            self.control.ground([("base", ())])
        if ground and not self.settings["quiet"]:
            end = time.process_time()
            print(f"done in {end-start:.1f}s")

    def configure_show(self):
        self.control.add("base", [], "#show." "#show remove/3.")

    def format_model(self, model):
        atoms = model.symbols(shown=True)
        return [py_of_symbol(atom) for atom in atoms]
