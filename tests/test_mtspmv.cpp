#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "mtSpMV.h"

#ifndef _OPENMP
#error "test_mtspmv must be compiled with OpenMP enabled"
#endif

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
  std::copy(x_values.begin(), x_values.end(), x.begin());

  spcraft::spmv_openmp<Ring>(A, x, y);
  if (!std::equal(y.begin(), y.end(), expected.begin())) {
    std::cerr << "OpenMP SpMV returned an incorrect result\n";
    return 1;
  }

  try {
    Vector wrong_size(A.n - 1);
    spcraft::spmv_openmp<Ring>(A, wrong_size, y);
    std::cerr << "OpenMP SpMV accepted an input vector with the wrong size\n";
    return 1;
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
