from optimo.symbolic import SymbolicFramework, SymbolContainer, SolverOutput
from optimo.symbolic import __variable_types, register_framework

import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    __variable_types.add(cp.Expression)
except ImportError:
    CVXPY_AVAILABLE = False


CVXPY = "cvxpy"


@register_framework(CVXPY)
class CvxpyModel(SymbolicFramework):

    def __init__(self):
        if not CVXPY_AVAILABLE:
            raise NameError("CVXPY could not be imported.")
        super().__init__()

    def _create_decision_var(self, name: str, shape: tuple, is_param: bool = False, **kwargs):
        return cp.Parameter(shape, name=name, **kwargs) if is_param else cp.Variable(shape, name=name, **kwargs)

    def init_constraints(self):
        return []

    def _add_inequality_constraint(self, expr, lower, upper):
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        nontrivial_lower = np.where(lower > -np.inf)[0]
        nontrivial_upper = np.where(upper < np.inf)[0]

        if len(nontrivial_upper) > 0:
            self.constraints.append(expr[nontrivial_upper] <= upper[nontrivial_upper])
        if len(nontrivial_lower) > 0:
            self.constraints.append(lower[nontrivial_lower] <= expr[nontrivial_lower])

    def _add_equality_constraint(self, expr, value):
        self.constraints.append(expr == value)

    def get_solver_success(self):
        if self.solver is not None:
            return self.solver.status not in ["infeasible", "unbounded"]
        return

    def get_solver_stats(self):
        return self.solver.solver_stats

    def __get_cost(self, num: bool = False):
        if self.cost is None:
            LOGGER.warn("Cost was not set! Using zero cost")
            f = 0
        else:
            f = self.cost
            if num:
                f = f.value
        return f

    def new_decision_var(self, name: str, shape: tuple, is_param: bool = False, is_aux_var: bool = False, lower=-np.inf, upper=np.inf):
        """Make a new decision variable and cache it.

        Args:
            name (str): Name of the variable.
            shape (tuple): shape of the variable
            is_param (bool): True if the new variable is used as a parameter (i.e., a numerical value for it will be provided when the optimization problem is called)
            is_aux_var (bool): When true, the variable is considered an auxiliary variable, so it can be saved in a separate cache. If ``is_param`` is True, this argument is ignored.

        Returns:
            The newly constructed variable
        """

        new_var = self._create_decision_var(name, shape, is_param)
        if is_param:
            self.params.add_variable(name, new_var, shape)
        else:
            lower = np.full(np.prod(shape), lower)
            upper = np.full(np.prod(shape), upper)
            self.add_inequality_constraint(new_var, lower, upper)
            if is_aux_var:
                self.aux_vars.add_variable(name, new_var, shape)
            else:
                self.vars.add_variable(name, new_var, shape)
        return new_var

    def get_u_from_decision_vars(self, solution):
        return solution

    def build(self):

        f = self.__get_cost()

        objective = cp.Minimize(f)
        problem = cp.Problem(objective=objective, constraints=self.constraints)

        self.solver = problem

    def _solve(self, param_values=SymbolContainer, initial_guess: np.ndarray = None):

        if len(self.params) > 0:
            for symb_param, num_param in zip(self.params.values(), param_values.values()):
                symb_param.value = num_param
        self.solver.solve(solver=cp.MOSEK)

        if not self.get_solver_success():
            # Otherwise, problem.value is inf or -inf, respectively.
            LOGGER.warn(f"Solver failed. Status: {self.solver.status}")

        minimizer = {name: var.value for name, var in self.vars.items()}
        return SolverOutput(self.solver.value, minimizer, self.get_solver_success(), self.get_solver_stats())

    # -----------------------------------------------------------
    # Operations
    # -----------------------------------------------------------

    @classmethod
    def concat(self, *arrays):
        return cp.hstack(arrays)
        # return cp.vstack(arrays)

    @classmethod
    def hstack(self, *arrays):
        return cp.hstack(arrays)

    @classmethod
    def vstack(self, *arrays):
        return cp.vstack(arrays)

    @classmethod
    def sum(self, array, *args, **kwargs):
        return cp.sum(array, *args, **kwargs)


# -----------------------------------------------------------
# /// END CVXPY
# -----------------------------------------------------------
