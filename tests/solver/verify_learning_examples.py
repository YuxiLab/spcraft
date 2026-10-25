"""Check the handbook's supplied arithmetic without implementing student LU."""

from fractions import Fraction as F
import unittest


def matrix(rows):
    return [[F(str(value)) for value in row] for row in rows]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def subtract(a, b):
    return [[x - y for x, y in zip(arow, brow)] for arow, brow in zip(a, b)]


class LearningExamples(unittest.TestCase):
    def test_scalar_factors_and_two_solves(self):
        a = matrix([[4, 1, 1], [2, 5, 0], [0, 1, 6]])
        lower = matrix([[1, 0, 0], ["1/2", 1, 0], [0, "2/9", 1]])
        upper = matrix([[4, 1, 1], [0, "9/2", "-1/2"], [0, 0, "55/9"]])
        x = matrix([[1], [2], [3]])
        rhs = matrix([[9], [12], [20]])
        y = matrix([[9], ["15/2"], ["55/3"]])
        self.assertEqual(multiply(lower, upper), a)
        self.assertEqual(multiply(a, x), rhs)
        self.assertEqual(multiply(lower, y), rhs)
        self.assertEqual(multiply(upper, x), y)

    def test_scalar_accumulator_trace(self):
        column1 = matrix([[1], [5], [1]])
        column2 = matrix([[1], [0], [6]])
        strict_l0 = matrix([[0], ["1/2"], [0]])
        self.assertEqual(subtract(column1, strict_l0), matrix([[1], ["9/2"], [1]]))
        after0 = subtract(column2, strict_l0)
        self.assertEqual(after0, matrix([[1], ["-1/2"], [6]]))
        strict_l1_times_u12 = matrix([[0], [0], ["-1/9"]])
        self.assertEqual(subtract(after0, strict_l1_times_u12),
                         matrix([[1], ["-1/2"], ["55/9"]]))

    def test_permutation(self):
        a = matrix([[4, 1, 1], [2, 5, 0], [0, 1, 6]])
        order = [2, 0, 1]
        b = [[a[i][j] for j in order] for i in order]
        self.assertEqual(b, matrix([[6, 0, 1], [1, 4, 1], [0, 2, 5]]))
        self.assertEqual(multiply(b, matrix([[3], [1], [2]])), matrix([[20], [9], [12]]))
        inverse = [order.index(i) for i in range(3)]
        self.assertEqual(inverse, [1, 2, 0])

    def test_dense_cross(self):
        lower = matrix([[1, 0], ["1/2", 1]])
        upper = matrix([[4, 2], [0, 3]])
        self.assertEqual(multiply(lower, upper), matrix([[4, 2], [2, 4]]))
        self.assertEqual(multiply(matrix([[1, 2]]), upper), matrix([[4, 8]]))
        self.assertEqual(multiply(lower, matrix([[2], [1]])), matrix([[2], [2]]))

    def test_nonsymmetric_factors_and_updates(self):
        b = matrix([[8, 0, 2, 0], [2, 10, "1/2", 3],
                    [0, 2, 12, "8/5"], [4, 0, 4, "65/4"]])
        lower = matrix([[1, 0, 0, 0], ["1/4", 1, 0, 0],
                        [0, "1/5", 1, 0], ["1/2", 0, "1/4", 1]])
        upper = matrix([[8, 0, 2, 0], [0, 10, 0, 3], [0, 0, 12, 1], [0, 0, 0, 16]])
        self.assertEqual(multiply(lower, upper), b)
        self.assertEqual(multiply(b, matrix([[1], [2], [3], [4]])),
                         matrix([[14], ["71/2"], ["232/5"], [81]]))
        original = matrix([[12, "8/5"], [4, "65/4"]])
        source0 = multiply(matrix([[0], ["1/2"]]), matrix([[2, 0]]))
        source1 = multiply(matrix([["1/5"], [0]]), matrix([[0, 3]]))
        self.assertEqual(subtract(subtract(original, source0), source1),
                         matrix([[12, 1], [3, "65/4"]]))
        self.assertEqual(multiply(matrix([[1, 0], ["1/4", 1]]),
                                  matrix([[12, 1], [0, 16]])),
                         matrix([[12, 1], [3, "65/4"]]))

    def test_all_three_update_products(self):
        lj = matrix([[1, 2], [0, 0]])
        lr = matrix([[3, 4], [5, 6]])
        uj = matrix([[2, 4], [1, 3]])
        uc = matrix([[6], [5]])
        self.assertEqual(subtract(matrix([[100, 100], [100, 100]]), multiply(lj, uj)),
                         matrix([[96, 90], [100, 100]]))
        self.assertEqual(subtract(matrix([[100, 100], [100, 100]]), multiply(lr, uj)),
                         matrix([[90, 76], [84, 62]]))
        self.assertEqual(subtract(matrix([[100], [100]]), multiply(lj, uc)),
                         matrix([[84], [100]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
