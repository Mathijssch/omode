from optimo.symbolic import SymbolicFramework, SymbolContainer, SolverOutput
from optimo.symbolic import __variable_types, register_framework
import numpy as np

from typing import Union, TYPE_CHECKING

import logging
LOGGER = logging.getLogger(__name__)

try:
    import casadi as cs
    CASADI_AVAILABLE = True
    __variable_types.add(cs.SX)
    __variable_types.add(cs.MX)


except ImportError:
    CASADI_AVAILABLE = False

CASADI = "casadi"

if TYPE_CHECKING:
    Vector = Union[np.ndarray, cs.SX, cs.MX]
    Variable = Vector


@register_framework(CASADI)
class CasadiModel(SymbolicFramework):

    nonconvex: bool = True

    def __init__(self, solver: str = "ipopt"):
        if not CASADI_AVAILABLE:
            raise NameError("CasADi could not be imported.")
        super().__init__(solver_name=solver)

    def _create_decision_var(self, name: str, shape: tuple, is_param: bool = False):
        return cs.SX.sym(name, *shape)

    def init_constraints(self):
        return {
            "g": [],
            "bounds": {
                "ubg": [],
                "lbg": [],
                "lbx": {
                    "vars": SymbolContainer("main_lbx"),
                    "aux": SymbolContainer("aux_lbx")
                },
                "ubx": {
                    "vars": SymbolContainer("main_ubx"),
                    "aux": SymbolContainer("aux_ubx")
                }
            }
        }

    @classmethod
    def _check_symbolic(cls, expr):
        assert isinstance(expr, (cs.SX, cs.MX)), f"Function {expr} is not a symbolic type"

    def _add_inequality_constraint(self, expr, lower, upper):
        self.constraints["g"].append(expr)
        self.constraints["bounds"]["lbg"].append(lower)
        self.constraints["bounds"]["ubg"].append(upper)

    def _add_equality_constraint(self, expr, value):
        self.constraints["g"].append(expr)
        self.constraints["bounds"]["lbg"].append(value)
        self.constraints["bounds"]["ubg"].append(value)

    # def get_cost(self, solution: SolverOutput):
    #     return super().get_cost(solution).full()

    def get_solver_success(self) -> bool:
        """Return True/False if the solver succeeded/failed. 
        If the solver has not been called yet, then return None. 
        """
        if self.solver is not None:
            return self.solver.stats()["success"]
        return

    def build(self):

        if self.cost is None:
            LOGGER.warn("Cost was not set! Using zero cost")
            f = 0
        else:
            f = self.cost

        x = self.concat(*self.flatten_vectors(self.vars.values()),
                        *self.flatten_vectors(self.aux_vars.values()))

        g = self.concat(*self.flatten_vectors(self.constraints["g"]))

        nlp = {"f": f, "g": g, "x": x}

        if len(self.params) > 0:
            nlp["p"] = self.concat(*self.flatten_vectors(self.params.values()))

        if self.solver_name in ["ipopt", "knitro"]:
            self.prepare_regular_casadi(nlp, solver=self.solver_name)
        # elif self.solver_name == "alpaqa":
        #     self.prepare_alpaqa(nlp)
        else:
            raise NotImplementedError(f"Solver {self.solver_name} is not supported!")

    def prepare_regular_casadi(self, nlp, *, solver: str = "ipopt"):
        if not self.verbose:
            self.solver_options = {"verbose_init": False, "print_time": False}
        if solver == "ipopt":
            self.solver_options.update({"ipopt": {"print_level": 0}})  # "tol": 1e-4}})

        if self.solver_options != {}:
            self.solver = cs.nlpsol("solver", solver, nlp, self.solver_options)
        else:
            self.solver = cs.nlpsol("solver", solver, nlp)

    def _flatten_constraint_bounds(self):
        return {
            "lbg": self.concat(*self.flatten_vectors(self.constraints["bounds"]["lbg"])),
            "ubg": self.concat(*self.flatten_vectors(self.constraints["bounds"]["ubg"])),
            "lbx": self.concat(
                *self.flatten_vectors(self.constraints["bounds"]["lbx"]["vars"].values()),
                *self.flatten_vectors(self.constraints["bounds"]["lbx"]["aux"].values())
            ),
            "ubx": self.concat(
                *self.flatten_vectors(self.constraints["bounds"]["ubx"]["vars"].values()),
                *self.flatten_vectors(self.constraints["bounds"]["ubx"]["aux"].values())
            ),
        }

    def _solve(self, param_values: SymbolContainer, initial_guess: Union[np.ndarray, list[np.ndarray], tuple[np.ndarray], dict[np.ndarray]]) -> SolverOutput:
        return self._solve_casadi_general(param_values, initial_guess)

    def _solve_casadi_general(self, param_values: SymbolContainer, initial_guess: Union[np.ndarray, list[np.ndarray], tuple[np.ndarray], dict[np.ndarray]]) -> SolverOutput:
        from time import perf_counter

        bounds = self._flatten_constraint_bounds()

        if len(self.params) > 0:
            param_values = self.concat(*self.flatten_vectors(param_values.values()))
            bounds["p"] = param_values

        if len(initial_guess) > 0:
            initial_guess = self.concat(*self.flatten_vectors(initial_guess.values()))
            bounds["x0"] = initial_guess

        start = perf_counter()
        sol = self.solver(**bounds)
        measured_time = perf_counter() - start
        self._solver_time = self.solver.stats().get('t_wall_total', measured_time)
        solver_success = self.get_solver_success()
        solution_vec = sol["x"]
        solution = {name: self.vars.extract_from_concatenation(solution_vec, name) for name in self.vars.keys()}
        return SolverOutput(sol["f"].full(), solution, solver_success, self.get_solver_stats(), )

    def get_solver_stats(self) -> dict:
        if self.solver is None:
            return dict()
        else:
            return self.solver.stats()

    # -----------------------------------------------------------
    # Operations
    # -----------------------------------------------------------

    @classmethod
    def scalar_product(cls, a, b):
        return cs.dot(a, b)

    @classmethod
    def vec(cls, v):
        return cs.vec(v)

    @classmethod
    def concat(self, *arrays):
        return cs.vertcat(*arrays)

    @classmethod
    def hstack(self, *arrays):
        return cs.horzcat(*arrays)

    @classmethod
    def vstack(self, *arrays):
        return cs.vertcat(*arrays)

    def l1_norm(self, a):
        return cs.sum(cs.abs(a))

    @classmethod
    def sum(cls, vector, *args, **kwargs):
        return cs.sum1(vector)

    @classmethod
    def sqsum(cls, vector):
        return cs.sumsqr(vector)
