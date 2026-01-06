from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple, List

import numpy as np

from simplex import lp_problem, math, pivoting_strategy


def compute_reduced_costs(
    problem: lp_problem.LpProblem,
    basic_variables: list[int],
    non_basic_variables: list[int],
    Binv: np.ndarray,
) -> np.ndarray:
    c: np.ndarray = problem.objective
    A: np.ndarray = problem.constraint_matrix
    return (
        c[non_basic_variables]
        - (c[basic_variables] @ Binv) @ A[:, non_basic_variables]
    )


class SolverStatus(Enum):
    SUCCESS = 0
    INFEASIBLE = 1
    UBOUNDED = 2

def compute_entering_variable_value(x: np.ndarray, u: np.ndarray) -> Tuple[int, float]:
    return min(((i, xi / ui) for i, (xi, ui) in enumerate(zip(x, u)) if ui > 0), key=lambda pair: pair[1])

# TODO: Add cycling detection, cycle proof pivoting strategy. Incorporate "pick_exiting_index" in pivot_strategy.
@dataclass
class Solver:
    pivoting_strategy_: pivoting_strategy.PivotingStrategy

    def find_basic_feasible_solution(
        self,
        problem: lp_problem.LpProblem,
    ) -> Tuple[SolverStatus, Optional[list[int]]]:
        """
        Finds a basic feasible solution by solving an auxiliary LP.
        """

        b: np.ndarray = problem.rhs.copy()
        A_original: np.ndarray = problem.constraint_matrix.copy()

        for i in range(len(b)):
            if b[i] < 0:
                A_original[i] *= -1
                b[i] *= -1

        A: np.ndarray = np.concatenate((A_original, np.eye(len(b))), axis=1)
        c: np.ndarray = np.concatenate((np.zeros(len(problem.objective)), np.ones(len(b))))
        
        feasibility_problem = lp_problem.LpProblem(A, b, c)

        B: list[int] = list(range(len(problem.objective), len(c)))
        success, (_, _, objective_value) = self.solve(feasibility_problem, B=B, log=False)

        if success != SolverStatus.SUCCESS:
            raise RuntimeError(
                "Could not solve auxiliary problem to find feasible starting point."
            )

        if objective_value != 0:
            return SolverStatus.INFEASIBLE, None
        return SolverStatus.SUCCESS, B

    def solve(
        self,
        problem: lp_problem.LpProblem,
        B: Optional[list[int]] = None,
        log: bool = False,
    ) -> Tuple[
        SolverStatus,
        Tuple[
            Optional[list[int]],
            Optional[list[float]],
            Optional[float],
        ],
    ]:
        if B is None:
            status, B = self.find_basic_feasible_solution(problem)
            if status == SolverStatus.INFEASIBLE:
                return status, (None, None, None)

        assert B is not None  # for type checkers

        A: np.ndarray = problem.constraint_matrix
        c: np.ndarray = problem.objective
        Binv: np.ndarray = np.linalg.inv(A[:, B])
        x: np.ndarray = Binv @ problem.rhs

        if log:
            print("Starting simplex algorithm...")

        iteration: int = 1
        while True:
            non_basic_variables: list[int] = [
                col for col in range(problem.variables) if col not in B
            ]

            reduced_costs = compute_reduced_costs(
                problem,
                B,
                non_basic_variables,
                Binv,
            )

            if np.all(reduced_costs >= 0):
                status = SolverStatus.SUCCESS
                break

            entering_index: int = non_basic_variables[self.pivoting_strategy_.pick_entering_index(reduced_costs)]
            u: np.ndarray = Binv @ A[:, entering_index]
            
            if np.all(u <= 0):
                status = SolverStatus.UBOUNDED
                break

            theta_index, theta_value = compute_entering_variable_value(x, u)
            exiting_index: int = B[theta_index]
            B[theta_index] = entering_index
            Binv = math.update_inverse(Binv, u, theta_index)

            x -= theta_value * u
            x[theta_index] = theta_value

            if log:
                print(
                    f"Iteration {iteration} ::: "
                    f"Leaving index: {exiting_index}, "
                    f"Entering index: {entering_index}, "
                    f"Objective: {c[B] @ x}"
                )
                
            iteration += 1

        if log:
            print(f"Simplex algorithm terminated after {iteration} iterations.")

        if status == SolverStatus.SUCCESS:
            solution: list[float] = [
                float(x[B.index(i)]) if i in B else 0.0
                for i in range(problem.variables)
            ]
            objective_value: float = float(c[B] @ x)
            return status, (B, solution, objective_value)

        return status, (None, None, None)