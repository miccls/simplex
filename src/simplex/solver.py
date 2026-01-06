from simplex import lp_problem, math
import numpy as np
# The outline for the simplex method.
# Given a problem in standard form, A, b, c
#
# min c^T x
#
# s.t
#   Ax = b
#   x >= 0
#
# Step 1: Find a feasible starting point.
#   Substep 1: ...
# Step 2: Take the feasible starting point and use the simplex method to solve it the problem
#
#
# For an iteration of the simplex algorithm, the following is inserted:
# B     - A vector of indices for the current solutions in the basis
# B_inv - The inverse of the basis matrix corresponding to B.
# x     - The associated basic feasible solution.

def compute_reduced_costs(problem, basic_variables, non_basic_variables, Binv):
    c = problem.objective
    A = problem.constraint_matrix
    return c[non_basic_variables] - (c[basic_variables] @ Binv) @ A[:, non_basic_variables]



class Solver:
    
    def __init__(self, pivoting_strategy):
        self.pivoting_strategy_ = pivoting_strategy

    def find_basic_feasible_solution(self, problem):
        # Finds a basic feasible solution by solving a separate LP
        # min y
        # s.t
        # Ax + y = b
        # x, y >= 0
        # This can be started with B = I
        # And the y's as basic variables.
        # The solution will be a feasible starting point if the objective is 0,
        # otherwise we conclude the LP is infeasible.
        
        b = problem.rhs.copy()
        A = problem.constraint_matrix.copy()

        for i in range(len(b)):
            if b[i] < 0:
                A[i] *= -1 
                b[i] *= -1 # Make rhs positive s.t y = b is valid initialization.
        A = np.concatenate((problem.constraint_matrix, np.eye(len(b))), axis = 1) # Add artificial variables.
        
        c = np.concatenate((np.zeros(len(problem.objective)), np.ones(len(b))))
        feasibility_problem = lp_problem.LpProblem(A, b, c)
        B = [i for i in range(len(problem.objective), len(c))]
        (success, (basis, _, _)) = self.solve(feasibility_problem, B = B, log = False)
        
        if not success:
            # Should not happen
            raise Exception("Could not solve auxiliary problem to find feasible starting point.")
        
        return basis
         

    def solve(self, problem: lp_problem.LpProblem, B = None, log = False):
        
        if B is None:
            B = self.find_basic_feasible_solution(problem)
        
        A = problem.constraint_matrix
        c = problem.objective
        
        Binv = np.linalg.inv(A[:, B])
        x = Binv @ problem.rhs
        
        terminate = False
        success = False # Make enum: success, unbounded, infeasible.
        
        iteration = 1
        
        if log:
            print("Starting simplex algoritm...")
        while not terminate:
            non_basic_variables = [column for column in range(problem.variables) if column not in B]
 
            # Compute the reduced costs
            reduced_costs = compute_reduced_costs(problem, B, non_basic_variables, Binv)
    
            # Check optimality
            if all(reduced_costs >= 0):
                terminate = True
                success = True
                continue
            
            # Choose one of the variables with negative reduced cost according to some (lexicographical) pivoting strategy
            within_base_index = self.pivoting_strategy_.pick_entering_index(reduced_costs)
            entering_index = non_basic_variables[within_base_index]
    
            # Compute change in basic vars from change in var corresponding to entering index
            u = Binv @ A[:, entering_index]
            
            # Check if problem is unbounded:
            if all(u <= 0):
                terminate = True
                continue # unbounded, optimal cost is -inf
            
            # Some component is positive, that means we can reduce objective!
            theta = min([(i,xi / ui) for i, (xi, ui) in enumerate(zip(x, u)) if ui > 0], key = lambda pair: pair[1])
            
            # Update the state accordingly
            exiting_index = B[theta[0]] 
            B[theta[0]] = entering_index
            Binv = math.update_inverse(Binv, u, theta[0])
            x -= theta[1] * u
            x[theta[0]] = theta[1]
            
            # Log iteration.
            if log:
                print(f"Iteration {iteration} ::: Leaving index: {exiting_index}, Entering index: {entering_index}, Objective: {c[B] @ x}")
            iteration += 1
        if log:
            print(f"Simplex algorithm terminated after {iteration} iterations.")
        
        solution = [x[B.index(i)] if i in B else 0 for i in range(problem.variables)]
        return (success, (B, solution, c[B] @ x))
    