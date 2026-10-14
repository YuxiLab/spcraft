"""Stage-level and end-to-end verification; no external packages required."""

from math import fsum
import random
import unittest

from sparse_solver import CSR, norm, poisson_2d
from sparse_lu import pivoting_example
from superlu_symbolic import (CSC, analyze, column_graph, minimum_degree,
                             symbolic_cholesky, tree_postorder)
from superlu_numeric import equilibrate
from superlu_pipeline import estimate_norm_1, factorize


def transpose_matvec(A, x):
    columns = [[] for _ in range(A.shape[1])]
    for i in range(A.shape[0]):
        for k in range(A.indptr[i], A.indptr[i + 1]):
            columns[A.indices[k]].append(A.data[k] * x[i])
    return [fsum(column) for column in columns]


class SymbolicTests(unittest.TestCase):
    def test_csc_conversion(self):
        A = CSR.from_coo((3, 3), [(2, 0, 4), (0, 1, 2), (1, 1, -3), (2, 2, 5)])
        csc = CSC.from_csr(A)
        self.assertEqual(csc.col_ptr, (0, 1, 3, 4))
        self.assertEqual(csc.row_indices, (2, 0, 1, 2))
        self.assertEqual(csc.data, (4, 2, -3, 5))
        self.assertEqual(csc.norm_1(), 5)

    def test_column_intersection_graph_and_etree(self):
        # Upper bidiagonal structure gives a path in the A^T*A graph.
        A = CSR.from_coo((4, 4), [(i, i, 2) for i in range(4)]
                        + [(i, i + 1, 1) for i in range(3)])
        graph = column_graph(A)
        self.assertEqual(graph, [{1}, {0, 2}, {1, 3}, {2}])
        etree, patterns = symbolic_cholesky(graph)
        self.assertEqual(etree, [1, 2, 3, 4])
        self.assertEqual(patterns, [{0, 1}, {1, 2}, {2, 3}, {3}])

    def test_minimum_degree_reduces_star_fill(self):
        graph = [{1, 2, 3}, {0}, {0}, {0}]
        order = minimum_degree(graph)
        self.assertEqual(order, [1, 2, 0, 3])
        _, natural = symbolic_cholesky(graph)
        inverse = {old: new for new, old in enumerate(order)}
        reordered = [{inverse[i] for i in graph[old]} for old in order]
        _, ordered = symbolic_cholesky(reordered)
        self.assertLess(sum(map(len, ordered)), sum(map(len, natural)))

    def test_postorder_forest_and_relaxed_subtrees(self):
        tree = [3, 3, 4, 4, 5, 6]
        post = tree_postorder(tree)
        self.assertEqual(sorted(post), list(range(6)))
        position = {node: i for i, node in enumerate(post)}
        for j, parent in enumerate(tree):
            if parent < 6:
                self.assertLess(position[j], position[parent])
        plan = analyze(poisson_2d(5), panel_size=3, relax=5)
        covered = [j for s, e in plan.panels for j in range(s, e)]
        self.assertEqual(covered, list(range(25)))
        self.assertTrue(all(1 <= e - s <= 3 for s, e in plan.panels))
        for s, e in plan.relaxed_groups:
            self.assertLessEqual(e - s, 5)
            root = e - 1
            for j in range(s, root):
                self.assertTrue(s <= plan.etree[j] <= root)

    def test_user_order_and_symbolic_reuse_validation(self):
        A = pivoting_example()
        plan = analyze(A, col_order=[2, 0, 1], postorder=False)
        self.assertEqual(plan.col_order, (2, 0, 1))
        changed = CSR(A.shape, tuple(v * 2 for v in A.data), A.indices, A.indptr)
        plan.check_pattern(changed)
        with self.assertRaises(ValueError):
            plan.check_pattern(CSR.from_coo((3, 3), [(i, i, 1) for i in range(3)]))
        for kwargs in ({"col_order": [0, 0, 2]}, {"panel_size": 0},
                       {"relax": -1}, {"ordering": "colamd"}):
            with self.assertRaises(ValueError):
                analyze(A, **kwargs)


class NumericTests(unittest.TestCase):
    def assert_factorization(self, solver):
        A, n = solver.scaled_A, solver.n
        original = [{} for _ in range(n)]
        for i in range(n):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                original[i][A.indices[k]] = A.data[k]
        for i in range(n):
            self.assertTrue(all(j < i for j in solver.L[i]))
            self.assertTrue(all(j >= i for j in solver.U[i]))
            for j in range(n):
                product = fsum((1.0 if i == k else solver.L[i].get(k, 0.0))
                               * solver.U[k].get(j, 0.0) for k in range(n))
                wanted = original[solver.row_order[i]].get(solver.analysis.col_order[j], 0.0)
                self.assertAlmostEqual(product, wanted, delta=1e-11 * max(1, abs(wanted)))

    def assert_solution(self, solver, expected, transpose=False, tolerance=1e-9):
        b = transpose_matvec(solver.A, expected) if transpose else solver.A.matvec(expected)
        actual = solver.solve(b, transpose=transpose)
        self.assertLess(norm([xi - ei for xi, ei in zip(actual, expected)]), tolerance)

    def test_equilibration_both_auto_none_and_original_scaling(self):
        # Dyadic entries keep the manufactured RHS exact despite large scaling.
        A = CSR.from_coo((2, 2), [(0, 0, 2.0 ** -20), (0, 1, 2.0 ** -32),
                                  (1, 0, 2.0 ** 10), (1, 1, 4)])
        scaled, scaling = equilibrate(A, "both")
        self.assertEqual(scaling.equed, "B")
        self.assertLessEqual(max(abs(v) for v in scaled.data), 1.0)
        for i in range(2):
            for k in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[k]
                self.assertEqual(scaled.data[k], A.data[k] * scaling.row[i] * scaling.column[j])
        self.assertEqual(equilibrate(A, "auto")[1].equed, "B")
        self.assertEqual(equilibrate(A, "none")[0], A)
        solver = factorize(A, equilibration="both", panel_size=2)
        self.assert_factorization(solver)
        self.assert_solution(solver, [2, -3])
        self.assert_solution(solver, [2, -3], transpose=True)

    def test_all_automatic_equilibration_decisions(self):
        cases = [
            ("N", [(0, 0, 4), (0, 1, 1), (1, 0, 1), (1, 1, 3)]),
            ("R", [(0, 0, 1e-6), (1, 1, 1e6)]),
            ("C", [(0, 0, 1), (0, 1, 1e-6), (1, 0, 0.5), (1, 1, 2e-6)]),
            ("B", [(0, 0, 2.0 ** -20), (0, 1, 2.0 ** -32),
                    (1, 0, 2.0 ** 10), (1, 1, 4)]),
        ]
        for equed, entries in cases:
            with self.subTest(equed=equed):
                solver = factorize(CSR.from_coo((2, 2), entries), equilibration="auto")
                self.assertEqual(solver.scaling.equed, equed)
                self.assert_solution(solver, [2, -3])
                self.assert_solution(solver, [2, -3], transpose=True)

    def test_late_pivot_and_column_permutation(self):
        A = pivoting_example()
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            for width in (1, 2, 3):
                with self.subTest(order=order, width=width):
                    plan = analyze(A, col_order=order, postorder=False, panel_size=width)
                    solver = factorize(A, analysis=plan, equilibration="none")
                    self.assert_factorization(solver)
                    self.assert_solution(solver, [1, 2, 3])
                    self.assert_solution(solver, [1, 2, 3], transpose=True)

    def test_threshold_pivoting(self):
        A = CSR.from_coo((2, 2), [(0, 0, 0.1), (0, 1, 1), (1, 0, 2), (1, 1, 3)])
        plan = analyze(A, ordering="natural", postorder=False)
        full = factorize(A, analysis=plan, equilibration="none", pivot_threshold=1)
        threshold = factorize(A, analysis=plan, equilibration="none", pivot_threshold=0.01)
        self.assertEqual(full.row_order[0], 1)
        self.assertEqual(threshold.row_order[0], 0)
        self.assert_solution(full, [1, 2])
        self.assert_solution(threshold, [1, 2])
        # A zero original diagonal is never accepted, including threshold=0.
        zero_diagonal = factorize(pivoting_example(), pivot_threshold=0, equilibration="none")
        self.assert_solution(zero_diagonal, [1, 2, 3])

    def test_block_updates_random_sparse_matrices(self):
        rng = random.Random(2026)
        for n in (4, 7, 12):
            entries = []
            for i in range(n):
                others = [(j, rng.uniform(-3, 3)) for j in range(n)
                          if j != i and rng.random() < 0.25]
                entries.extend((i, j, v) for j, v in others)
                entries.append((i, i, 1 + sum(abs(v) for _, v in others)))
            base = CSR.from_coo((n, n), entries)
            order = list(range(n))
            rng.shuffle(order)
            A = CSR.from_coo((n, n), [(i, base.indices[k], base.data[k])
                                    for i, old in enumerate(order)
                                    for k in range(base.indptr[old], base.indptr[old + 1])])
            expected = [rng.uniform(-2, 2) for _ in range(n)]
            for width in (1, 2, 5):
                for ordering in ("natural", "minimum_degree"):
                    with self.subTest(n=n, width=width, ordering=ordering):
                        solver = factorize(A, panel_size=width, ordering=ordering)
                        self.assert_factorization(solver)
                        self.assert_solution(solver, expected)
                        self.assert_solution(solver, expected, transpose=True)

    def test_dense_supernodes_used_in_solves(self):
        A = CSR.from_coo((4, 4), [(i, j, 8.0 if i == j else 1.0 + (i + j) / 10)
                                  for i in range(4) for j in range(4)])
        solver = factorize(A, panel_size=4)
        self.assertEqual(len(solver.supernodes), 1)
        self.assertEqual((solver.supernodes[0].start, solver.supernodes[0].end), (0, 4))
        self.assertEqual(solver.U_csc.col_ptr[-1], solver.nnz_U)
        self.assertEqual(solver.numeric_stats.current_work_nnz, solver.nnz_U)
        self.assertGreaterEqual(solver.numeric_stats.peak_work_nnz, solver.nnz_U)
        self.assert_solution(solver, [1, -2, 3, -4])
        self.assert_solution(solver, [1, -2, 3, -4], transpose=True)

    def test_ordering_reduces_poisson_factor_storage(self):
        A = poisson_2d(8)
        natural = factorize(A, ordering="natural", postorder=False)
        ordered = factorize(A, ordering="minimum_degree")
        self.assertLess(ordered.nnz_L + ordered.nnz_U, natural.nnz_L + natural.nnz_U)
        self.assert_solution(ordered, [float(i % 5) for i in range(64)])

    def test_multiple_rhs_and_same_pattern_refactor(self):
        A = poisson_2d(4)
        solver = factorize(A)
        vectors = [[1.0] * 16, [float(i) for i in range(16)]]
        result = solver.solve_many([A.matvec(x) for x in vectors])
        for expected, actual in zip(vectors, result):
            self.assertLess(norm([a - e for a, e in zip(actual, expected)]), 1e-10)
        changed = CSR(A.shape, tuple(2 * v for v in A.data), A.indices, A.indptr)
        refactored = solver.refactor(changed)
        self.assertIs(refactored.analysis, solver.analysis)
        self.assertTrue(refactored.symbolic_reused)
        self.assertEqual(refactored.numeric_stats.reused_pivots, 16)
        self.assert_solution(refactored, vectors[1])
        self.assert_solution(solver, vectors[1])  # Original factors remain usable.
        with self.assertRaises(ValueError):
            solver.refactor(CSR.from_coo((16, 16), [(i, i, 1) for i in range(16)]))

    def test_refactor_rejects_unsafe_old_pivots(self):
        A = CSR.from_coo((2, 2), [(0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 3)])
        plan = analyze(A, ordering="natural", postorder=False)
        solver = factorize(A, analysis=plan, equilibration="none")
        changed = CSR.from_coo((2, 2), [(0, 0, 0.01), (0, 1, 1), (1, 0, 2), (1, 1, 3)])
        updated = solver.refactor(changed)
        self.assertEqual(updated.row_order, [1, 0])
        self.assert_solution(updated, [1, 2])

    def test_empty_scalar_singular_and_invalid_inputs(self):
        empty = factorize(CSR.from_coo((0, 0), []))
        self.assertEqual(empty.solve([]), [])
        self.assertEqual(empty.refine([]).reciprocal_condition_estimate, 1)
        scalar = factorize(CSR.from_coo((1, 1), [(0, 0, -2)]))
        self.assertEqual(scalar.solve([6]), [-3])
        for A in [CSR.from_coo((2, 2), [(0, 0, 1)]),
                  CSR.from_coo((2, 2), [(0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 4)])]:
            with self.assertRaises(ValueError):
                factorize(A)
        for kwargs in ({"pivot_threshold": -1}, {"pivot_threshold": 1.1},
                       {"pivot_atol": -1}, {"equilibration": "bad"}):
            with self.assertRaises(ValueError):
                factorize(pivoting_example(), **kwargs)
        with self.assertRaises(ValueError):
            scalar.solve([float("nan")])
        with self.assertRaises(ValueError):
            scalar.solve([1, 2])


class DiagnosticsTests(unittest.TestCase):
    def test_norm_estimate_and_original_condition_after_scaling(self):
        diagonal = [1.0, 0.5, 0.1]
        op = lambda x: [d * v for d, v in zip(diagonal, x)]
        self.assertAlmostEqual(estimate_norm_1(3, op, op), 1)
        A = CSR.from_coo((3, 3), [(0, 0, 1), (1, 1, 2), (2, 2, 10)])
        solver = factorize(A, equilibration="both")
        self.assertAlmostEqual(solver.reciprocal_condition_estimate(), 0.1)
        self.assertEqual(solver.reciprocal_pivot_growth(), 1)

    def test_refinement_improves_perturbed_solution(self):
        A = poisson_2d(5)
        solver = factorize(A)
        expected = [float(i % 7) for i in range(25)]
        b = A.matvec(expected)
        report = solver.refine(b, [xi + 1e-3 for xi in expected])
        self.assertGreaterEqual(report.refinement_steps, 1)
        self.assertLess(report.backward_error_history[-1], report.backward_error_history[0] * 1e-8)
        self.assertLess(norm([x - e for x, e in zip(report.x, expected)]), 1e-10)
        self.assertLess(report.relative_residual, 1e-13)
        self.assertLess(report.componentwise_backward_error, 1e-13)
        self.assertGreaterEqual(report.forward_error_estimate, 0)

    def test_transpose_refinement_and_double_residual(self):
        solver = factorize(pivoting_example())
        expected = [1.0, -2.0, 3.0]
        b = transpose_matvec(solver.A, expected)
        report = solver.refine(b, [xi + 0.1 for xi in expected], transpose=True,
                               residual_precision="double")
        self.assertLess(norm([x - e for x, e in zip(report.x, expected)]), 1e-12)
        self.assertLess(report.componentwise_backward_error, 1e-13)
        self.assertTrue(0 < report.reciprocal_condition_estimate <= 1)

    def test_zero_rhs_and_refinement_iteration_limit(self):
        solver = factorize(poisson_2d(2))
        report = solver.refine([0] * 4)
        self.assertEqual(report.relative_residual, 0)
        self.assertEqual(report.componentwise_backward_error, 0)
        self.assertEqual(report.forward_error_estimate, 0)
        report = solver.refine([1] * 4, [0] * 4, maxiter=0)
        self.assertEqual(report.refinement_steps, 0)
        self.assertEqual(report.componentwise_backward_error, 1)
        for kwargs in ({"maxiter": -1}, {"residual_precision": "bad"}):
            with self.assertRaises(ValueError):
                solver.refine([1] * 4, **kwargs)


if __name__ == "__main__":
    unittest.main()
