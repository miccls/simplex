from abc import ABC, abstractmethod
from typing import override

import jaxtyping
import numpy as np
from common.lp_problem import LpProblem
from common.numpy_type_aliases import ArrayF, ArrayI

PIVOTING_TOLERANCE = 1e-5


class PrimalPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        """
        Selects the variable that should enter the basis.

        Args:
            reduced_costs: reduced costs
            non_basic_vars: variable indices of the non basic variables, assumed to be sorted
        Returns:
            The variable index, i.e. an element of the array `non_basic_vars`, of a variable that should enter the basis.
        """
        ...

    @abstractmethod
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        Selects the index exiting the basis.

        If we compare the arguments with "Numerical Optimization", Nocedal & Wright, page 370,
        we can make the following identifications:

        `x_basis[p]` is the value for the decision variable `x_k`, where `k=basis[p]`.

        `basic_direction` is `d`, with `d = B^-1 * A_q`, where B is the basis matrix `A[:, basis]`,
        and `A_q = A[:, q]` is the column of the constraint matrix for the entering variable `x_q`.

        Args:
            basis: Variable indices for the basic variables.
            x_basis: Values for the basic variables.
            basic_direction: The basic direction for the entering variable.

        Returns:
            Index `p` in ``basis`` array for the decision variable that should be removed from the basis.
        """
        ...


class DualPivotingStrategy(ABC):
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        """
        Gives stateful pivoting strategies a chance to reset for a new LP/basis.
        Stateless rules intentionally leave this as a no-op.
        """
        del problem, basis

    @abstractmethod
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        """
        TODO(martins): Describe purpose of picking entering index
        """
        ...

    @abstractmethod
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        s: jaxtyping.Float[ArrayF, " num_nonbasic"],
        pivot_direction: jaxtyping.Float[ArrayF, " num_nonbasic"],
    ) -> int:
        """
        TODO(martins): Describe purpose of picking exiting index
        """
        ...


def index_of_smallest_ratio(
    basis: jaxtyping.Int[ArrayI, " m"],
    x_basis: jaxtyping.Float[ArrayF, " m"],
    basic_direction: jaxtyping.Float[ArrayF, " m"],
) -> int:
    """
    Args:
        basis: Variable indices for the basic variables.
        x_basis: Values for the basic variables.
        basic_direction: The basic direction for the entering variable.

    Returns:
        Index `i` in `basis` array with the smallest positive ratio `x_basis[i] / basic_direction[i]`,
        choosing the index corresponding to the lowest variable index `basis[i]` in case of ties.
    """
    # TODO(you): Implement the minimum-ratio test.
    return -1


class BlandsRule(PrimalPivotingStrategy):
    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to Bland's rule.
        return -1

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index according to Bland's rule.
        return -1


class DantzigsRule(PrimalPivotingStrategy):
    """Dantzig's rule is one of the simplest pivoting strategies. It was
    suggested by George Dantzig, inventor of the Primal Simplex algorithm.
    It simply selectes the variable with the most negative reduced cost.

    See section 13.5 in "Numerical Optimization" for more details.

    Since all rules use smallest subscript for the exiting index, that is not tested
    """

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to Dantzig's rule.
        return -1

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index according to Dantzig's rule.
        return -1


class SteepestEdgeRule(PrimalPivotingStrategy):
    """Primal steepest-edge pricing for the entering variable."""

    def __init__(
        self,
        problem: LpProblem | None = None,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> None:
        self.problem: LpProblem | None = None
        self.entering_index = -1
        self.non_basic_vars = np.array([], dtype=int)
        self.norm_eta_squared = np.array([], dtype=float)

        if problem is not None and initial_basis is not None:
            self.initialize(problem, initial_basis)

    @override
    def initialize(
        self,
        problem: LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
    ) -> None:
        # TODO(you): Initialize steepest-edge weights for a new LP/basis.
        del basis
        self.problem = problem
        self.entering_index = -1
        self.non_basic_vars = np.array([], dtype=int)
        self.norm_eta_squared = np.array([], dtype=float)

    def _update_eta(
        self,
        exiting_index: int,
        basis: jaxtyping.Int[ArrayI, " m"],
        b_inv: jaxtyping.Float[ArrayF, "m m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> None:
        # TODO(you): Update steepest-edge weights after a pivot.
        del exiting_index, basis, b_inv, basic_direction

    @override
    def pick_entering_index(
        self,
        reduced_costs: jaxtyping.Float[ArrayF, " num_nonbasic"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
    ) -> int:
        # TODO(you): Pick entering index according to the steepest-edge rule.
        return -1

    @override
    def pick_exiting_index(
        self,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        basic_direction: jaxtyping.Float[ArrayF, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index and update steepest-edge weights.
        return -1


class DualBlandsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index according to Bland's rule for the dual simplex.
        return -1

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        # TODO(you): Pick entering index according to Bland's rule for the dual simplex.
        return -1


class DualDantzigsRule(DualPivotingStrategy):
    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index according to Dantzig's rule for the dual simplex.
        return -1

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        # TODO(you): Pick entering index according to Dantzig's rule for the dual simplex.
        return -1


class DualSteepestEdgeRule(DualPivotingStrategy):
    """Dual steepest-edge leaving-row rule."""

    @override
    def pick_exiting_index(
        self,
        primal_vars: jaxtyping.Float[ArrayF, " m"],
        basic_vars: jaxtyping.Int[ArrayI, " m"],
        inv_basis_matrix: jaxtyping.Float[ArrayF, "m m"] | None = None,
    ) -> int:
        # TODO(you): Pick exiting index according to the dual steepest-edge rule.
        return -1

    @override
    def pick_entering_index(
        self,
        non_basic_vars: jaxtyping.Int[ArrayI, " m"],
        s: jaxtyping.Float[ArrayF, " m"],
        pivot_direction: jaxtyping.Float[ArrayF, " m"],
    ) -> int:
        # TODO(you): Pick entering index according to the dual steepest-edge rule.
        return -1
