#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "SpCraft.h"

namespace
{

int failures = 0;

void CheckText(std::string_view got, std::string_view expected, std::string_view what)
{
  if (got != expected) {
    std::cerr << what << ": expected\n" << expected << "got\n" << got << '\n';
    ++failures;
  }
}

template <class Exception, class Action>
void CheckThrows(Action action, std::string_view what)
{
  try {
    action();
    std::cerr << what << ": expected an exception, got none\n";
    ++failures;
  } catch (const Exception&) {
  } catch (const std::exception& error) {
    std::cerr << what << ": wrong exception: " << error.what() << '\n';
    ++failures;
  }
}

template <class Matrix>
void CheckSparse(const Matrix& matrix, std::string_view format)
{
  std::ostringstream out;
  spcraft::PrintMatrix(matrix, 1, 1, 2, 3, "A", out);
  CheckText(out.str(), "A (4 x 5), rows [1, 3), cols [1, 4):\n  1 2 3\n1 0 5 .\n2 . . 8\n", format);

  out.str("");
  spcraft::PrintMatrix(matrix, 2, 3, std::numeric_limits<std::int64_t>::max(), 10, "edge", out);
  CheckText(out.str(), "edge (4 x 5), rows [2, 4), cols [3, 5):\n  3 4\n2 8 .\n3 . .\n",
            "clip oversized subblock without overflowing");

  out.str("");
  spcraft::PrintMatrix(matrix, 2, 2, "top", out);
  CheckText(out.str(), "top (4 x 5), rows [0, 2), cols [0, 2):\n  0 1\n0 . .\n1 . 0\n",
            "top-left shorthand");

  out.str("");
  spcraft::PrintMatrix(matrix, 4, 5, 1, 1, "empty", out);
  CheckText(out.str(), "empty (4 x 5), rows [4, 4), cols [5, 5):\n[]\n", "empty edge block");
  out.str("");
  spcraft::PrintMatrix(matrix, 1, 1, 0, 2, "zero", out);
  CheckText(out.str(), "zero (4 x 5), rows [1, 1), cols [1, 3):\n[]\n", "zero row count");

  CheckThrows<std::invalid_argument>([&] { spcraft::PrintMatrix(matrix, -1, 0, 1, 1, "bad", out); },
                                     "negative row");
  CheckThrows<std::invalid_argument>([&] { spcraft::PrintMatrix(matrix, 0, 0, 1, -1, "bad", out); },
                                     "negative count");
  CheckThrows<std::out_of_range>([&] { spcraft::PrintMatrix(matrix, 5, 0, 1, 1, "bad", out); },
                                 "row outside matrix");
  CheckThrows<std::out_of_range>([&] { spcraft::PrintMatrix(matrix, 0, 6, 1, 1, "bad", out); },
                                 "column outside matrix");
}

template <class Matrix>
void CheckEigen(const Matrix& matrix)
{
  std::ostringstream actual, native;
  spcraft::PrintMatrix(matrix, 1, 1, 2, 3, "E", actual);
  native << "E (4 x 5), rows [1, 3), cols [1, 4):\n" << matrix.block(1, 1, 2, 3) << '\n';
  CheckText(actual.str(), native.str(), "Eigen native block formatting");
  actual.str("");
  native.str("");
  spcraft::PrintMatrix(matrix, 3, 4, 10, 10, "edge", actual);
  native << "edge (4 x 5), rows [3, 4), cols [4, 5):\n" << matrix.block(3, 4, 1, 1) << '\n';
  CheckText(actual.str(), native.str(), "Eigen edge clipping");
  CheckThrows<std::out_of_range>([&] { spcraft::PrintMatrix(matrix, 0, 6, 1, 1, "bad", actual); },
                                 "Eigen bounds check");
}

template <class Matrix>
void CheckDuplicates(const Matrix& matrix)
{
  std::ostringstream out;
  spcraft::PrintMatrix(matrix, 1, 1, "duplicates", out);
  if (out.str().find("{2, 3, 4}") == std::string::npos) {
    std::cerr << "Expected duplicate entries {2, 3, 4}, got\n" << out.str();
    ++failures;
  }
}

}  // namespace

int main()
{
  spcraft::CooMatrix<int, double, std::int64_t> coo;
  coo.Allocate(5, 4, 5);
  coo.entries[0] = {2, 3, 8};
  coo.entries[1] = {0, 4, 9};
  coo.entries[2] = {1, 1, 0};
  coo.entries[3] = {3, 0, 7};
  coo.entries[4] = {1, 2, 5};
  const auto csr = coo.ToCsr();
  const auto csc = coo.ToCsc();
  CheckSparse(coo, "unsorted COO subblock");
  CheckSparse(csr, "CSR subblock");
  CheckSparse(csc, "CSC subblock");

  spcraft::CooMatrix<int, int> duplicates;
  duplicates.Allocate(3, 1, 1);
  duplicates.entries[0] = {0, 0, 2};
  duplicates.entries[1] = {0, 0, 3};
  duplicates.entries[2] = {0, 0, 4};
  CheckDuplicates(duplicates);
  CheckDuplicates(duplicates.ToCsr());
  CheckDuplicates(duplicates.ToCsc());

  std::ostringstream out;
  spcraft::PrintMatrix(spcraft::CsrMatrix<int, double>{}, 5, 5, "empty", out);
  CheckText(out.str(), "empty (0 x 0), rows [0, 0), cols [0, 0):\n[]\n", "empty CSR");
  out.str("");
  spcraft::CscMatrix<unsigned, bool> empty;
  empty.Allocate(0, 2, 3);
  spcraft::PrintMatrix(empty, 1, 2, 1, 1, "empty shaped", out);
  CheckText(out.str(), "empty shaped (2 x 3), rows [1, 2), cols [2, 3):\n  2\n1 .\n",
            "unsigned indices and an empty matrix with shape");

  Eigen::MatrixXd eigen(4, 5);
  eigen << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
  CheckEigen(eigen);
  CheckEigen(eigen + eigen);
  CheckEigen(eigen.block(0, 0, 4, 5));
  Eigen::SparseMatrix<double> sparse = eigen.sparseView();
  sparse.coeffRef(1, 2) = 0;
  sparse.prune(0.0);
  CheckEigen(sparse);
  Eigen::SparseMatrix<double, Eigen::RowMajor> row_major = sparse;
  CheckEigen(row_major);
  out.str("");
  spcraft::PrintMatrix(Eigen::MatrixXd{}, 5, 5, "empty", out);
  CheckText(out.str(), "empty (0 x 0), rows [0, 0), cols [0, 0):\n[]\n", "empty Eigen");

  if (failures != 0) {
    std::cerr << failures << " matrix printing check(s) failed\n";
    return 1;
  }
  std::cout << "PrintMatrix checks passed\n";
  return 0;
}
