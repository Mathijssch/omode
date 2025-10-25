from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Union
from collections import OrderedDict

__variable_types = set([np.ndarray])


LOGGER = logging.getLogger(__name__)


class DimensionMismatchError(Exception):
    """Exception for mismatching shapes of arrays"""

    def __init__(self, expected=None, got=None, message=""):
        msg = f"Dimension mismatch: {message}"
        if expected is not None:
            msg += f"expected {expected}"
            if got is not None:
                msg += f" - got {got})."
        super().__init__(msg)


class NonConvexException(Exception):
    def __init__(self, name: str):
        super().__init__(
            f"Requested nonconvex solver, but {name} does not support nonconvex optimization."
        )


@dataclass
class SolverOutput:
    optimal_cost: float
    minimizer: np.ndarray
    success: bool
    stats: dict
    message: str = ""


@dataclass
class SymbolContainer:
    name: str = "container"
    __variables: OrderedDict = field(default_factory=OrderedDict, init=False)
    __shapes: OrderedDict = field(default_factory=OrderedDict, init=False)
    __curr_index: int = field(default_factory=lambda: 0, init=False)

    @property
    def variables(self):
        return self.__variables

    @property
    def shapes(self):
        return self.__shapes

    def add_variable(self, name: str, value, shape: tuple[int] = None):
        if name in self.variables:
            raise NameExistsException(name, self)

        self.__variables[name] = value
        if shape is None:
            shape = value.shape
        self._update_shapes(name, shape)

    def items(self):
        return self.__variables.items()

    def keys(self):
        return self.__variables.keys()

    def values(self):
        return self.__variables.values()

    def __getitem__(self, idx: str):
        return self.__variables[idx]

    def __len__(self):
        return len(self.__variables)

    def __contains__(self, key: str):
        return key in self.__variables

    def _update_shapes(self, name, shape):
        index_offset = self.__update_shapes_array_helper(
            self.__shapes, self.__curr_index, name, shape
        )
        self.__curr_index += index_offset

    @classmethod
    def __update_shapes_array_helper(
        cls, shape_dict: dict, curr_idx: int, name: str, shape: Tuple[int]
    ) -> int:
        """Add the current running index and the given shape to the shape dictionary and return the total number of elements added.

        Args:
            shape_dict (dict): shape dictionary to update
            curr_idx (int): running index to start from
            name (str): name of the variable to record the shape for
            shape (Tuple[int]): shape of the new variable

        Returns:
            int
        """
        shape_dict[name] = (curr_idx, shape)
        return np.prod(shape)

    def __align_iterable(self, other: list | tuple):
        new = SymbolContainer(f"{self.name}_values")
        for i, (name, expected_shape) in enumerate(self.__shapes.items()):
            expected_shape = expected_shape[
                1
            ]  # First component is the starting index of the variable.
            try:
                new_value = other[i]
            except IndexError:
                raise ValueError(
                    f"Parameter container does not contain sufficiently many entries. Got {len(other)}. Expected {len(self)}."
                )

            if new_value.shape != expected_shape:
                raise DimensionMismatchError(
                    expected_shape,
                    new_value.shape,
                    f"Shape mismatch for parameter {name}. ",
                )
            new.add_variable(name, new_value, new_value.shape)
        return new

    def __align_dict(self, other: Union[dict, "SymbolContainer"]):
        name = (
            f"{self.name}_values"
            if not isinstance(other, SymbolContainer)
            else other.name
        )
        new = SymbolContainer(name)
        for name, expected_shape in self.__shapes.items():
            expected_shape = expected_shape[
                1
            ]  # First component is the starting index of the variable.
            try:
                new_value = other[name]
            except KeyError:
                raise ValueError(
                    f"Parameter container does not contain an entry for the parameter {name}."
                )

            if new_value.shape != expected_shape:
                raise DimensionMismatchError(
                    expected_shape,
                    new_value.shape,
                    f"Shape mismatch for parameter {name}. ",
                )
            new.add_variable(name, new_value, new_value.shape)
        return new

    def align(
        self, other: Union[tuple, list, dict, "SymbolContainer"]
    ) -> "SymbolContainer":
        """Align a given container with ``self``.

        Map the variables in ``other`` to the symbols contained in ``self`` and return a new SymbolContainer which has the same structure as self (contains the same variables in the same order), but contains the given quantities.

        Args:
            other (Union[tuple, list, dict, SymbolContainer]): Data to add to the new SymbolContainer

        Raises:
            RuntimeError: _description_
            KeyError: _description_
            DimensionMismatchError: _description_
            DimensionMismatchError: _description_
            NameError: _description_
            NameError: _description_

        Returns:
            SymbolContainer
        """

        if isinstance(other, (list, tuple)):
            return self.__align_iterable(other)
        if isinstance(other, (dict, SymbolContainer)):
            return self.__align_dict(other)

    def vec_to_dict(self, vector: np.ndarray):
        return {
            name: self.extract_from_concatenation(vector, name) for name in self.keys()
        }

    def extract_from_concatenation(self, vector: np.ndarray, name: str):
        if name not in self.__shapes:
            raise KeyError(f"Variable with name {name} does not exist in {self.name}")

        idx, shape = self.__shapes[name]
        total_size = np.prod(shape)
        return np.reshape(vector[idx : idx + total_size], shape, order="F")

    def __str__(self):
        desc = [f"{n} {v.shape} (type: {type(v)})" for n, v in self.variables.items()]
        return f"SymbolContainer {self.name}.\nVariables:\n {'; '.join(desc)}"


class NameExistsException(ValueError):
    def __init__(self, name: str, container: SymbolContainer, *args: object) -> None:
        super().__init__(*args)
        self.name = name
        self.container = container

    def __str__(self) -> str:
        return f"The name {self.name} for a {self.container.name} is already taken. "


class SymbolicFramework(ABC):
    name: str = "abstract"
    nonconvex: bool = False

    def __init__(self, solver_name: str = None):
        self.vars = SymbolContainer("vars")
        self.aux_vars = SymbolContainer("aux_vars")
        self.params = SymbolContainer("params")
        self.constraints = self.init_constraints()
        self.cost = None
        self.solver_name = solver_name
        self._solver_time = None
        self.solver = None
        self.verbose = False
        self.solver_options = dict()

    def get_or_make_param(self, name: str, shape: tuple[int]):
        if name in self.params:
            par = self.params[name]
            existing_shape = self.params.shapes[name][1]
            if existing_shape != shape:
                raise DimensionMismatchError(
                    f"Requested parameter with name `{name}` and shape {shape}. The existing parameter with this name has shape {existing_shape}."
                )
            return par
        else:
            return self.new_decision_var(name, shape, is_param=True)

    @abstractmethod
    def _create_decision_var(self, name: str, shape: tuple, is_param: bool): ...

    def new_decision_var(
        self,
        name: str,
        shape: tuple,
        is_param: bool = False,
        is_aux_var: bool = False,
        lower=-np.inf,
        upper=np.inf,
    ):
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
        lbx = np.full(shape, lower)
        ubx = np.full(shape, upper)

        if is_param:
            self.params.add_variable(name, new_var, shape)
        elif is_aux_var:
            self.aux_vars.add_variable(name, new_var, shape)
            self.constraints["bounds"]["lbx"]["aux"].add_variable(name, lbx, shape)
            self.constraints["bounds"]["ubx"]["aux"].add_variable(name, ubx, shape)
        else:
            self.vars.add_variable(name, new_var, shape)
            self.constraints["bounds"]["lbx"]["vars"].add_variable(name, lbx, shape)
            self.constraints["bounds"]["ubx"]["vars"].add_variable(name, ubx, shape)
        return new_var

    def get_cost(self, solution: SolverOutput):
        return solution.optimal_cost

    @abstractmethod
    def get_solver_success(self) -> bool:
        """Return True/False if the solver succeeded/failed.
        If the solver has not been called yet, then return None.
        """
        ...

    @property
    def solver_time(self) -> float:
        return self._solver_time

    def solve(
        self,
        param_values: Union[
            list[np.ndarray], tuple[np.ndarray], dict[np.ndarray], SymbolContainer
        ] = None,
        initial_guess: Union[
            list[np.ndarray], tuple[np.ndarray], dict[np.ndarray], SymbolContainer
        ] = None,
    ) -> SolverOutput:
        if self.solver is None:
            raise RuntimeError(
                "Solver must be built first. See ``SymbolicFramework.build``"
            )

        if param_values is None:
            param_values = []

        if initial_guess is None:
            initial_guess = []

        param_values = self.params.align(param_values)

        return self._solve(param_values, initial_guess)

    @abstractmethod
    def _solve(self, param_values, initial_guess) -> SolverOutput: ...

    def get_u_from_decision_vars(self, solution):
        u_symb = self.concat(*self.vars.values())
        size = np.prod(self.shape(u_symb))
        return solution[:size]

    def get_numerical_decision_variable(self, solution: np.ndarray, name: str):
        return self.vars.extract_from_concatenation(solution, name)

    def get_solver_stats(self) -> dict:
        return dict()

    def _preprocess_constraint(self, expr, value):
        self._check_symbolic(expr)
        value = self._vectorize_number(value, self.shape(expr))
        self._check_shape(expr, value)
        return value

    def add_inequality_constraint(self, expr, lower, upper):
        lower = self._preprocess_constraint(expr, lower)
        upper = self._preprocess_constraint(expr, upper)
        self._add_inequality_constraint(expr, lower, upper)

    def add_equality_constraint(self, expr, value):
        value = self._preprocess_constraint(expr, value)
        self._add_equality_constraint(expr, value)

    @abstractmethod
    def _add_inequality_constraint(self, expr, lower, upper):
        """Add an inequality constraint assuming that the expression is symbolic and the shapes of `expr` match with `lower` and `upper`.

        Args:
            expr: Expression to be constrained
            lower: Element-wise lower bound on the expression
            upper: Element-wise upper bound on the expression
        """
        ...

    @abstractmethod
    def _add_equality_constraint(self, expr, value):
        """Add an inequality constraint assuming that the expression is symbolic and the shapes of `expr` match with `lower` and `upper`.

        Args:
            expr: Expression to be constrained
            value: Value the expression must take
        """
        ...

    @classmethod
    def _vectorize_number(cls, value, shape):
        if isinstance(value, (int, float)):
            value = float(value) * np.ones(shape).squeeze()
        value = value.reshape(shape)
        return value

    @classmethod
    def _check_shape(cls, v1, v2):
        shape1 = cls.shape(v1)
        shape2 = cls.shape(v2)
        if len(shape1) != len(shape2):
            raise DimensionMismatchError(shape1, shape2, "Lengths of shapes differ.")
        if not all(s1 == s2 for s1, s2 in zip(shape1, shape2)):
            raise DimensionMismatchError(shape1, shape2)

    @classmethod
    def _check_symbolic(cls, expr):
        pass

    def set_cost(self, cost):
        self.cost = cost

    @abstractmethod
    def init_constraints(self): ...

    @abstractmethod
    def build():
        pass

    @classmethod
    def scalar_product(cls, a, b):
        return a.T @ b

    @classmethod
    def quadform(cls, x, Q):
        return cls.scalar_product(x, cls.matvec(Q, x))

    @classmethod
    def shape(cls, a):
        return a.shape

    @classmethod
    def size(cls, a):
        return np.prod(cls.shape(a))

    @classmethod
    def cos(cls, a):
        return NotImplementedError()

    @classmethod
    def sin(cls, a):
        return NotImplementedError()

    @classmethod
    def concat(cls, a):
        raise NotImplementedError()

    @classmethod
    def hstack(cls, a):
        raise NotImplementedError()

    @classmethod
    def vstack(cls, a):
        raise NotImplementedError()

    @classmethod
    def matvec(cls, a, b):
        return a @ b

    @classmethod
    def vec(cls, vec):
        raise NotImplementedError()

    @classmethod
    def sum(cls, vector, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def flatten_vectors(cls, vecs: Iterable):
        """reshape all vectors in `vecs` to  (n x 1).

        Args:
            vecs (Iterable): Iterable of expressions
        """
        return [cls.vec(v) for v in vecs]

    def flatten_decision_vars(
        self, vars: SymbolContainer | dict = None, aux: SymbolContainer | dict = None
    ):
        vars = self.vars if vars is None else self.vars.align(vars)
        aux_vars = self.aux_vars if aux is None else self.aux_vars.align(aux)

        return self.concat(
            *self.flatten_vectors(vars.values()),
            *self.flatten_vectors(aux_vars.values()),
        )

    @classmethod
    def l1_norm(cls, a):
        return np.sum(np.abs(a))

    @classmethod
    def sqsum(cls, a):
        """Return the squared norm of a vector"""
        raise NotImplementedError()

    @classmethod
    def bmat(cls, arrays):
        raise NotImplementedError()

    @classmethod
    def rot_z(cls, x):
        s = cls.sin
        c = cls.cos

        return cls.bmat(((c(x), -s(x), 0.0), (s(x), c(x), 0.0), (0.0, 0.0, 1.0)))

    @classmethod
    def rot_y(cls, x):
        s = cls.sin
        c = cls.cos

        return cls.bmat(((c(x), 0, s(x)), (0, 1, 0.0), (-s(x), 0.0, c(x))))

    @classmethod
    def rot_x(cls, x):
        s = cls.sin
        c = cls.cos

        return cls.bmat(((1, 0.0, 0.0), (0, c(x), -s(x)), (0.0, s(x), c(x))))

    @classmethod
    def rot_matrix(cls, angles):
        Rx = cls.rot_x(angles[0])
        Ry = cls.rot_y(angles[1])
        Rz = cls.rot_z(angles[2])
        return Rz @ Ry @ Rx


FRAMEWORKS: dict[str, SymbolicFramework] = {}


def register_framework(name: str):
    def decorator(f):
        FRAMEWORKS[name] = f
        f.name = name
        return f

    return decorator


class NoSuchFrameworkException(Exception):
    def __init__(self, f: str):
        super().__init__(
            f"The symbolic framework {f} is not available. Expected one of ({', '.join(FRAMEWORKS.keys())})"
        )


def get_framework(name: str, nonconvex: bool = False, **opts) -> SymbolicFramework:
    """Get the framework with the given name.
    If `nonconvex` is True, then we check that the provided framework can
    model nonconvex optimization problems. If it cannot, we throw an error.
    """
    try:
        framework = FRAMEWORKS[name](**opts)
        if nonconvex and not framework.nonconvex:
            raise NonConvexException(name)
        return framework
    except KeyError:
        raise NoSuchFrameworkException(name)
