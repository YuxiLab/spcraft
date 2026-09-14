#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "SpCraft.h"

int main()
{
  using Matrix = spcraft::CsrMatrix<std::int32_t, double, std::int64_t>;
  using Ring = spcraft::PlusTimesRing<double>;
  using Vector = spcraft::DenseVector<std::int32_t, double>;

  // [ 2  0 -1  0  0.5 ]
  // [ 0  0  0  0  0   ]  (empty row)
  // [ 0  3  0 -2  0   ]
  // [-4  0  0  0  1   ]
  Matrix A;
  A.Allocate(7, 4, 5);

  const std::array<std::int64_t, 5> row_ptr{0, 3, 3, 5, 7};
  const std::array<std::int32_t, 7> col_id{0, 2, 4, 1, 3, 0, 4};
  const std::array<double, 7> values{2.0, -1.0, 0.5, 3.0, -2.0, -4.0, 1.0};
  for (std::size_t i = 0; i < row_ptr.size(); ++i) {
    A.row_ptr[i] = row_ptr[i];
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    A.col_id[i] = col_id[i];
    A.val[i] = values[i];
  }

  const std::array<double, 5> x_values{1.0, 2.0, -3.0, 4.0, 5.0};
  const std::array<double, 4> expected{7.5, 0.0, -2.0, 1.0};
  Vector x(static_cast<std::int32_t>(x_values.size()));
  Vector y(static_cast<std::int32_t>(expected.size()));
  std::copy(x_values.begin(), x_values.end(), x.val);

  spcraft::OmpSpMV<Ring>(A, x, y);
  if (!std::equal(expected.begin(), expected.end(), y.val)) {
    std::cerr << "OpenMP SpMV returned an incorrect result\n";
    return 1;
  }

  try {
    Vector wrong_size(A.n - 1);
    spcraft::OmpSpMV<Ring>(A, wrong_size, y);
    std::cerr << "OpenMP SpMV accepted an input vector with the wrong size\n";
    return 1;
  } catch (const std::invalid_argument&) {
  }

  // sparse_left = [ 1 0 2 ]
  //               [ 0 3 0 ]
  Matrix sparse_left;
  sparse_left.Allocate(3, 2, 3);
  const std::array<std::int64_t, 3> left_row_ptr{0, 2, 3};
  const std::array<std::int32_t, 3> left_col_id{0, 2, 1};
  const std::array<double, 3> left_values{1.0, 2.0, 3.0};
  std::copy(left_row_ptr.begin(), left_row_ptr.end(), sparse_left.row_ptr);
  std::copy(left_col_id.begin(), left_col_id.end(), sparse_left.col_id);
  std::copy(left_values.begin(), left_values.end(), sparse_left.val);

  // dense_right = [ 4 0 ]
  //               [ 0 5 ]
  //               [ 6 7 ]
  const std::array<double, 6> dense_right{4.0, 0.0, 0.0, 5.0, 6.0, 7.0};
  std::array<double, 4> dense_result{};
  const std::array<double, 4> expected_dense{16.0, 14.0, 0.0, 15.0};
  spcraft::spmm_openmp<Ring>(sparse_left, dense_right.data(), dense_result.data(), 2);
  if (dense_result != expected_dense) {
    std::cerr << "OpenMP SpMM returned an incorrect result\n";
    return 1;
  }

  // Sample left_dense * transpose(right_dense) at sparse_left's pattern.
  const std::array<double, 4> left_dense{1.0, 2.0, 3.0, 4.0};
  const std::array<double, 6> right_dense{5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  const std::array<double, 3> expected_samples{17.0, 29.0, 53.0};
  spcraft::sddmm_openmp<Ring>(sparse_left, left_dense.data(), right_dense.data(), 2);
  if (!std::equal(expected_samples.begin(), expected_samples.end(), sparse_left.val)) {
    std::cerr << "OpenMP SDDMM returned an incorrect result\n";
    return 1;
  }

  return 0;
}
