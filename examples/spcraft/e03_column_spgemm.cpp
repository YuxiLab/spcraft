#include <iostream>

#include "SpCraft.h"

// Prepare canonical DCSC inputs, then compute native COO output with OmpHashSpGEMM.
int main()
{
  using Ring = spcraft::PlusTimesRing<double>;
  using CSC = spcraft::CscMatrix<int, double>;
  using DCSC = spcraft::DcscMatrix<int, double>;

  // A = [1 0 2]     B = [4 0]
  //     [0 3 0]         [0 5]
  //                     [6 7]
  // These stack arrays are borrowed: the pointer constructors do not own them.
  int a_ptr[] = {0, 1, 2, 3}, a_rows[] = {0, 1, 0};
  double a_values[] = {1, 3, 2};
  int b_ptr[] = {0, 2, 4}, b_rows[] = {0, 2, 1, 2};
  double b_values[] = {4, 6, 5, 7};
  const CSC a(a_ptr, a_rows, a_values, 3, 2, 3);
  const CSC b(b_ptr, b_rows, b_values, 4, 3, 2);
  // Conversion copies inputs and drops empty column headers before multiplication.
  const auto da = DCSC::FromCsc(a), db = DCSC::FromCsc(b);
  const auto c = spcraft::OmpHashSpGEMM<Ring>(da, db);
  for (int p = 0; p < c.nnz; ++p) {
    const auto& [row, column, value] = c.entries[p];
    std::cout << '(' << row << ", " << column << ") = " << value << '\n';
  }
  // The kernel emits COO directly; convert explicitly when a consumer needs CSR.
  const auto csr = c.ToCsr();
  std::cout << "CSR nonzeros: " << csr.nnz << '\n';
}
