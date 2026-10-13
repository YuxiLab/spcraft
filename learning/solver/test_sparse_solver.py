"""Run with: python3 -m unittest discover -s sparse_learning -v"""

import unittest

from sparse_solver import CSR, conjugate_gradient, norm, poisson_2d


class SparseSolverTests(unittest.TestCase):
    def test_coo_duplicates_cancellation_and_empty_row(self):
        A = CSR.from_coo((3, 3), [(2, 1, 5), (0, 2, 3), (0, 2, -1),
                                  (0, 0, 7), (0, 0, -7)])
        self.assertEqual(A.indptr, (0, 1, 1, 2))
        self.assertEqual(A.indices, (2, 1))
        self.assertEqual(A.matvec([2, 3, 4]), [8, 0, 15])

    def test_hand_computed_solution(self):
        # [4 1; 1 3] * [1 2] = [6 7].
        A = CSR.from_coo((2, 2), [(0, 0, 4), (0, 1, 1),
                                  (1, 0, 1), (1, 1, 3)])
        for jacobi in (False, True):
            result = conjugate_gradient(A, [6, 7], jacobi=jacobi)
            self.assertTrue(result.converged)
            self.assertLessEqual(result.iterations, 2)
            self.assertAlmostEqual(result.x[0], 1)
            self.assertAlmostEqual(result.x[1], 2)

    def test_poisson_manufactured_solution(self):
        A = poisson_2d(8)
        expected = [1 + (i % 7) / 7 for i in range(64)]
        b = A.matvec(expected)
        for jacobi in (False, True):
            result = conjugate_gradient(A, b, jacobi=jacobi, rtol=1e-11)
            self.assertTrue(result.converged)
            self.assertLess(norm([x - e for x, e in zip(result.x, expected)]), 1e-8)
            actual = norm([bi - ai for bi, ai in zip(b, A.matvec(result.x))])
            self.assertLessEqual(actual, 1e-11 * norm(b))
            self.assertEqual(actual, result.residual_norms[-1])

    def test_jacobi_solves_ill_scaled_diagonal_in_one_step(self):
        A = CSR.from_coo((3, 3), [(0, 0, 1), (1, 1, 100), (2, 2, 10000)])
        result = conjugate_gradient(A, [1, 200, 30000], jacobi=True)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.x, [1, 2, 3])

    def test_zero_rhs_and_exact_initial_guess(self):
        A = poisson_2d(2)
        for b, x0 in [([0] * 4, None), (A.matvec([1] * 4), [1] * 4)]:
            result = conjugate_gradient(A, b, x0=x0)
            self.assertTrue(result.converged)
            self.assertEqual(result.iterations, 0)

    def test_absolute_tolerance_and_iteration_limit(self):
        A = poisson_2d(2)
        result = conjugate_gradient(A, [1e-20] * 4, atol=1e-19)
        self.assertTrue(result.converged)
        result = conjugate_gradient(A, [1] * 4, maxiter=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    def test_invalid_inputs_and_breakdown(self):
        with self.assertRaises(ValueError):
            CSR.from_coo((2, 2), [(2, 0, 1)])
        with self.assertRaises(ValueError):
            CSR((2, 2), (1,), (0,), (0, 2, 1))
        with self.assertRaises(ValueError):
            conjugate_gradient(poisson_2d(2), [1])
        with self.assertRaises(ValueError):
            conjugate_gradient(poisson_2d(2), [1] * 4, rtol=-1)
        with self.assertRaises(ValueError):
            conjugate_gradient(poisson_2d(2), [float("nan")] * 4)
        negative = CSR.from_coo((1, 1), [(0, 0, -1)])
        with self.assertRaises(ValueError):
            conjugate_gradient(negative, [1])
        with self.assertRaises(ValueError):
            conjugate_gradient(negative, [1], jacobi=True)


if __name__ == "__main__":
    unittest.main()
