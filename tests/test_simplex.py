import unittest

from simplex import pivoting_strategy, math, lp_problem, solver

import numpy as np

class TestPivoting(unittest.TestCase):

    def setUp(self):
        self.smallest_subscript_rule = pivoting_strategy.SmallestSubscriptRule()

    def test_smallest_subscript(self):
        reduced_costs = [1, 2, 3, 4, -1, -2, -3]
        self.assertEqual(self.smallest_subscript_rule.pick_entering_index(reduced_costs), 4) 
        
    def test_throws_correctly(self):
        reduced_costs = [1]
        with self.assertRaises(Exception) as e:
            self.smallest_subscript_rule.pick_entering_index(reduced_costs)
        self.assertEqual(str(e.exception), "Pivoting strategy assumes some reduced cost is negative.")
        
class TestInverseComputation(unittest.TestCase):
    
    def test_update_inverse(self):
        A = np.array([
             [1,2,3,4],
             [4,3,2,1],
            ])
        basis = [1, 2]
        basis_matrix = A[:, basis]
        self.assertTrue((basis_matrix == [[2,3],[3,2]]).all())
        Binv = np.linalg.inv(basis_matrix)
        
        # Switch out index 2 for index three
        entering_index = 3
        exiting_index = 2
        new_basis = [1,3]
        
        d = Binv @ A[:, entering_index]
        Binv = math.update_inverse(Binv, d, basis.index(exiting_index))
        
        self.assertTrue(np.allclose(Binv, np.linalg.inv(A[:, new_basis])))
        
class TestSolver(unittest.TestCase):
    
    def test_problem_with_start_solution(self):
        # Example 3.5 in "Introduction to Linear Programming", page 101.    
        A = np.array([
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ])
        
        b = np.array([20, 20, 20]).T
        
        c = np.array([-10, -12, -12, 0, 0, 0])
        
        # Starting basis.
        B = [3,4,5]
        
        Binv = np.linalg.inv(A[:, B])
        x = Binv @ b
        
        # Assert feasibility of starting point.
        self.assertTrue(np.allclose(A[:, B] @ x, b))
        self.assertTrue((x >= 0).all())
        
        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        
        test_problem = lp_problem.LpProblem(A, b, c)
        (success, (basis, solution, objective_value)) = simplex_solver.solve(test_problem, B = B)
        
        # Check we have solved the problem correctly!
        self.assertTrue(success)
        self.assertTrue(all(i in [0,1,2] for i in basis))
        self.assertTrue(np.allclose(solution, [4,4,4,0,0,0]))
        self.assertEqual(-136, objective_value)

    def test_problem_without_start_solution(self):
        # Example 3.5 in "Introduction to Linear Programming", page 101.    
        A = np.array([
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ])
        
        b = np.array([20, 20, 20]).T
        
        c = np.array([-10, -12, -12, 0, 0, 0])
        
        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        
        test_problem = lp_problem.LpProblem(A, b, c)
        (success, (basis, solution, objective_value)) = simplex_solver.solve(test_problem)
        
        # Check we have solved the problem correctly!
        self.assertTrue(success)
        self.assertTrue(all(i in [0,1,2] for i in basis))
        self.assertTrue(np.allclose(solution, [4,4,4,0,0,0]))
        self.assertAlmostEqual(-136, objective_value)

if __name__ == "__main__":
    unittest.main()