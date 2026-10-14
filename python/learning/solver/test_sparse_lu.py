"""Check factorization identities, pivoting, fill-in, and reusable solves."""

import random
import unittest

from sparse_lu import pivoting_example, sparse_lu
from sparse_solver import CSR, norm, poisson_2d


class SparseLUTests(unittest.TestCase):
    def assert_factorization(self, A, factors):
        # Independently check every entry of P*A=L*U for these small problems.
        n = A.shape[0]
        original = [{} for _ in range(n)]
        for i in range(n):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[k]
                original[i][j] = original[i].get(j, 0) + A.data[k]
        for i in range(n):
            self.assertTrue(all(j < i for j in factors.L[i]))
            self.assertTrue(all(j >= i for j in factors.U[i]))
            for j in range(n):
                product = sum(
                    (1.0 if i == k else factors.L[i].get(k, 0.0))
                    * factors.U[k].get(j, 0.0)
                    for k in range(n)
                )
                self.assertAlmostEqual(product, original[factors.perm[i]].get(j, 0),
                                       places=10)

    def test_initial_and_late_pivoting(self):
        A = pivoting_example()
        factors = sparse_lu(A)
        self.assertEqual(factors.perm, [2, 0, 1])
        self.assert_factorization(A, factors)
        self.assertEqual(factors.solve([7, 3, 11]), [1, 2, 3])

    def test_fill_in(self):
        A = CSR.from_coo((3, 3), [(0, 0, 4), (0, 1, 1), (0, 2, 1),
                                  (1, 0, 1), (1, 1, 4),
                                  (2, 0, 1), (2, 2, 4)])
        factors = sparse_lu(A)
        self.assertAlmostEqual(factors.U[1][2], -0.25)
        self.assertAlmostEqual(factors.L[2][1], -1 / 15)
        self.assertGreater(factors.nnz_L + factors.nnz_U - 3, factors.input_nnz)
        self.assert_factorization(A, factors)

    def test_reuse_factors_and_preserve_input(self):
        A = poisson_2d(6)
        before = (A.data, A.indices, A.indptr)
        factors = sparse_lu(A)
        factor_before = ([row.copy() for row in factors.L],
                         [row.copy() for row in factors.U], factors.perm.copy())
        for expected in ([1.0] * 36, [float(i % 7 - 3) for i in range(36)]):
            b = A.matvec(expected)
            x = factors.solve(b)
            self.assertLess(norm([xi - ei for xi, ei in zip(x, expected)]), 1e-10)
            self.assertLess(norm([bi - ai for bi, ai in zip(b, A.matvec(x))]), 1e-10)
        self.assertEqual((A.data, A.indices, A.indptr), before)
        self.assertEqual((factors.L, factors.U, factors.perm), factor_before)

    def test_nonsymmetric_random_sparse_matrices(self):
        rng = random.Random(42)
        for n in (2, 5, 9):
            entries = []
            for i in range(n):
                off_diagonal = [(j, rng.uniform(-2, 2))
                                for j in range(n) if j != i and rng.random() < 0.3]
                entries.extend((i, j, value) for j, value in off_diagonal)
                entries.append((i, i, 1 + sum(abs(v) for _, v in off_diagonal)))
            base = CSR.from_coo((n, n), entries)
            # Shuffle equation rows to force pivoting while retaining invertibility.
            order = list(range(n))
            rng.shuffle(order)
            A = CSR.from_coo((n, n), [
                (i, base.indices[k], base.data[k])
                for i, old_i in enumerate(order)
                for k in range(base.indptr[old_i], base.indptr[old_i + 1])
            ])
            factors = sparse_lu(A)
            self.assert_factorization(A, factors)
            expected = [rng.uniform(-3, 3) for _ in range(n)]
            x = factors.solve(A.matvec(expected))
            self.assertLess(norm([xi - ei for xi, ei in zip(x, expected)]), 1e-10)

    def test_exact_cancellation(self):
        A = CSR.from_coo((3, 3), [(0, 0, 2), (0, 1, 2),
                                  (1, 0, 1), (1, 1, 1), (1, 2, 1),
                                  (2, 1, 1), (2, 2, 1)])
        factors = sparse_lu(A)
        self.assert_factorization(A, factors)
        self.assertTrue(all(v != 0 for row in factors.U for v in row.values()))

    def test_empty_and_one_by_one(self):
        self.assertEqual(sparse_lu(CSR.from_coo((0, 0), [])).solve([]), [])
        self.assertEqual(sparse_lu(CSR.from_coo((1, 1), [(0, 0, -2)])).solve([6]), [-3])

    def test_scale_and_duplicate_entries(self):
        tiny = CSR.from_coo((1, 1), [(0, 0, 1e-100)])
        self.assertEqual(sparse_lu(tiny).solve([2e-100]), [2])
        with self.assertRaises(ValueError):
            sparse_lu(tiny, pivot_atol=1e-90)
        duplicates = CSR((1, 1), (1.0, 2.0), (0, 0), (0, 2))
        self.assertEqual(sparse_lu(duplicates).solve([6]), [2])

    def test_singular_and_invalid_inputs(self):
        for A in [CSR.from_coo((2, 2), [(0, 0, 1), (0, 1, 2),
                                        (1, 0, 2), (1, 1, 4)]),
                  CSR.from_coo((2, 2), [(0, 0, 1)])]:
            with self.assertRaises(ValueError):
                sparse_lu(A)
        with self.assertRaises(ValueError):
            sparse_lu(CSR.from_coo((2, 3), []))
        with self.assertRaises(ValueError):
            sparse_lu(pivoting_example(), pivot_atol=-1)
        factors = sparse_lu(pivoting_example())
        with self.assertRaises(ValueError):
            factors.solve([1])
        with self.assertRaises(ValueError):
            factors.solve([1, 2, float("nan")])


if __name__ == "__main__":
    unittest.main()
