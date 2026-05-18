import logging
import time

import jaxtyping
import numpy as np
import scipy.linalg
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

import simplex_util
from simplex import linear_algebra, pivoting_strategy
from simplex_util import (
    INVERSE_RECOMPUTE_INTERVAL,
    NON_NEGATIVITY_TOLERANCE,
    SolveHistory,
    get_non_basic_vars,
)

logger = logging.getLogger(__name__)
LOG_FIRST_ITERATIONS = 10
LOG_INTERVAL = 100


class DualSimplex:
    pivoting_strategy_: pivoting_strategy.DualPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.DualPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.DualBlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def _setup_artificial_problem(
        self, problem: lp_problem.LpProblem, big_m: float = 1e6
    ) -> tuple[
        lp_problem.LpProblem,
        jaxtyping.Int[ArrayI, " m"],
    ]:
        """
        Sets up the problem for a Phase 1 Dual Simplex by adding a single artificial constraint.

        This method handles problems where an initial basis is not dual-feasible by adding one
        artificial constraint and one artificial variable. This allows us to perform a "magic pivot"
        that immediately yields a dual-feasible basis.

        Mathematical Formulation:
        1. Start with an arbitrary basis B and compute its reduced costs s.
        2. If some s_k < 0, the basis is not dual-feasible.
        3. Augment the primal problem with a new constraint:
           \\sum_{j \\notin B} x_j + x_{art} = M
           where x_{art} is an artificial slack variable with cost 0, and M is a large scalar.
        4. The initial basis for this augmented problem is B U {x_{art}}.
        5. We force x_{art} to exit the basis and x_k (where k = argmin(s)) to enter. This is the "magic pivot".
           Because the artificial row has a coefficient of 1 for all non-basic variables, pivoting
           on it subtracts s_k from all reduced costs: s_j' = s_j - s_k.
           Since s_k is the minimum reduced cost, s_j' >= 0 for all j, making the new basis
           B U {x_k} strictly dual-feasible!

        For reference, this is commonly known as the "Single Artificial Constraint" method
        for the dual simplex method.
        """
        # TODO(you): Implement the setup for the artificial problem to find a dual-feasible basis.
        # Use a rank-revealing QR factorization to find an initial independent basis.
        m, _ = problem.constraint_matrix.shape
        scipy.linalg.qr(problem.constraint_matrix, pivoting=True)
        return problem, np.zeros(m, dtype=int)

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        is_augmented: bool = False,
        original_num_variables: int = 0,
    ) -> simplex_util.SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        final_basis = basis

        # TODO(you): Handle the augmented problem case if necessary.

        return simplex_util.SolveResult(
            basis=final_basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> simplex_util.SolveResult:
        self.solve_history_ = SolveHistory()

        original_num_variables = problem.num_variables
        is_augmented = False

        if initial_basis is None:
            problem, initial_basis = self._setup_artificial_problem(problem)
            is_augmented = problem.num_variables > original_num_variables

        basis = initial_basis
        non_basic_vars = get_non_basic_vars(problem.num_variables, basis)

        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
        self.pivoting_strategy_.initialize(problem, basis)

        # TODO(you): Compute the basic variable values.
        x_basis = np.zeros(len(basis))

        logger.info("Starting Dual Simplex algorithm...")
        self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
        logger.info(
            f"Initial objective value {self.solve_history_.objective_history[-1]}"
        )

        logger.info("Iter     Objective      Primal Inf.    Dual Inf.    Time")
        start = time.time()

        for iteration in range(1, max_iterations):
            # TODO(you): Check for dual-simplex optimality.
            if np.all(x_basis >= -NON_NEGATIVITY_TOLERANCE):
                logger.info(
                    f"Simplex algorithm found optimal objective {self.solve_history_.objective_history[-1]} after {iteration - 1} iterations."
                )
                return self._finalize_result(
                    problem, basis, x_basis, is_augmented, original_num_variables
                )

            exiting_index = self.pivoting_strategy_.pick_exiting_index(
                x_basis, basis, inv_basis_matrix
            )

            # TODO(you): Compute reduced costs and the dual pivot direction.
            s_non_basic = np.zeros(len(non_basic_vars))
            non_basic_direction = np.zeros(len(non_basic_vars))

            if np.max(non_basic_direction) <= pivoting_strategy.PIVOTING_TOLERANCE:
                raise simplex_util.InfeasibleLpError(
                    "Dual problem is unbounded, therefore Primal is infeasible."
                )

            entering_index = self.pivoting_strategy_.pick_entering_index(
                non_basic_vars, s_non_basic, non_basic_direction
            )
            entering_variable = non_basic_vars[entering_index]

            # TODO(you): Update the basic solution from the dual-simplex direction.
            x_basis = np.zeros(len(basis))

            # TODO(you): Keep the non-basic variables aligned with the basis update.
            non_basic_vars[entering_index] = basis[exiting_index]
            basis[exiting_index] = entering_variable

            # TODO(you): Update the inverse of the basis matrix.
            if iteration % INVERSE_RECOMPUTE_INTERVAL == 0:
                inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
            else:
                inv_basis_matrix = linear_algebra.update_inverse(
                    problem.constraint_matrix,
                    inv_basis_matrix,
                    int(entering_variable),
                    int(exiting_index),
                )

            self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))

            if (iteration < LOG_FIRST_ITERATIONS) or (iteration % LOG_INTERVAL == 0):
                logger.info(
                    f"{iteration:4d}    {problem.objective[basis].T @ x_basis:10.3e}     "
                    f"{np.sum(np.abs(problem.constraint_matrix[:, basis] @ x_basis - problem.rhs)) - np.sum(np.minimum(x_basis, 0.0)):10.3e}     {abs(min(np.min(s_non_basic), 0.0)):10.3e}"
                    f"    {time.time() - start:.4}s"
                )

        raise simplex_util.IterationLimitError
