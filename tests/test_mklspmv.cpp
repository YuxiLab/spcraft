#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "SpCraft.h"

template <class NT>
bool test_mkl_spmv()
{
  using Index = std::int32_t;
  spcraft::CsrMatrix<Index, NT, Index> matrix;
  matrix.Allocate(6, 3, 3);

  const Index row_ptr[] = {0, 2, 4, 6};
  const Index col_id[] = {0, 2, 0, 1, 1, 2};
  const NT values[] = {2, 1, -1, 3, 4, 5};
  std::copy(std::begin(row_ptr), std::end(row_ptr), matrix.row_ptr);
  std::copy(std::begin(col_id), std::end(col_id), matrix.col_id);
  std::copy(std::begin(values), std::end(values), matrix.val);

  const std::vector<NT> x = {1, 2, 3};
  std::vector<NT> actual(3);
  const std::vector<NT> expected = {5, 5, 23};

  spcraft::MklCsrSpmv<Index, NT, Index> mkl_matrix(matrix, 10);
  mkl_matrix.Multiply(x.data(), actual.data());

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > static_cast<NT>(1.0e-6)) {
      std::cerr << "oneMKL SpMV mismatch at row " << i << ": got " << actual[i]
                << ", expected " << expected[i] << '\n';
      return false;
    }
  }
  return true;
}

int main()
{
  return test_mkl_spmv<float>() && test_mkl_spmv<double>() ? 0 : 1;
}
