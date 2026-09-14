#include <iostream>

#include "SpCraft.h"

int main()
{
  using COO = spcraft::CooMatrix<int, double>;
  COO matrix;
  matrix.Allocate(3, 2, 3);
  // One raw Entry array stores (row, column, value), in any insertion order.
  matrix.entries[0] = {1, 2, 3.0};
  matrix.entries[1] = {0, 1, 2.0};
  matrix.entries[2] = {1, 0, 4.0};

  // The pointer constructor borrows live entries; the owner remains matrix.
  COO view(matrix.entries, matrix.nnz, matrix.m, matrix.n);
  const auto copy = view.Clone();
  for (int i = 0; i < copy.nnz; ++i) {
    const auto& [row, column, value] = copy.entries[i];
    std::cout << row << ' ' << column << ' ' << value << '\n';
  }

  // Conversion groups entries by row without changing their values.
  const auto csr = matrix.ToCsr();
  std::cout << "CSR nonzeros: " << csr.nnz << '\n';
}
