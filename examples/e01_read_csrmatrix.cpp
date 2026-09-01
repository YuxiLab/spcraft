#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "spcraft.h"

using namespace spcraft;

using SpCsr = CsrMatrix<int32_t, float>;
using SpCoo = CooMatrix<int32_t, float>;

void PrintMatrix(SpCsr& A)
{
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    fmt::print(stderr,
               "Error: Expected exactly 1 argument (matrix file path).\nUsage: {} "
               "<matrix_file.mtx>\n",
               argv[0]);
    return 1;
  }

  std::string mtx_path = argv[1];
  const int kMaxPrint = 10;

  try {
    fmt::print("\nReading Matrix Market file in COO format: {}\n", mtx_path);
    spcraft::CooMatrix<int, double> coo =
        spcraft::CooMatrix<int, double>::FromMatrixMarket(mtx_path);

    fmt::print("\n=== COO Matrix Summary ===\n");
    fmt::print("Rows (m): {}, Cols (n): {}, Nonzeros (nnz): {}\n", coo.m, coo.n, coo.nnz);

    int print_count = std::min<int>(coo.nnz, kMaxPrint);
    if (print_count > 0) {
      fmt::print("Showing first {} entries:\n", print_count);
      for (int i = 0; i < print_count; ++i) {
        fmt::print("  [{}] row: {}, col: {}, val: {}\n", i, coo.row_id[i], coo.col_id[i],
                   coo.val[i]);
      }
      if (coo.nnz > kMaxPrint) {
        fmt::print("  ... ({} entries omitted)\n", coo.nnz - kMaxPrint);
      }
    }

    fmt::print("\nConverting COO to CSR...\n");
    spcraft::CsrMatrix<int, double> csr = coo.ToCsr();

    fmt::print("\n=== CSR Matrix Summary ===\n");
    fmt::print("Rows (m): {}, Cols (n): {}, Nonzeros (nnz): {}\n", csr.m, csr.n, csr.nnz);

    int print_row_ptrs = std::min<int>(csr.m + 1, kMaxPrint);
    std::vector<int> row_ptr_slice(csr.row_ptr, csr.row_ptr + print_row_ptrs);
    fmt::print("Row pointers (row_ptr, first {}): {}\n", print_row_ptrs, row_ptr_slice);
    if (csr.m + 1 > kMaxPrint) {
      fmt::print("  ... ({} row pointers omitted)\n", (csr.m + 1) - kMaxPrint);
    }

    int print_nnz = std::min<int>(csr.nnz, kMaxPrint);
    std::vector<int> col_id_slice(csr.col_id, csr.col_id + print_nnz);
    std::vector<double> val_slice(csr.val, csr.val + print_nnz);

    fmt::print("Column indices (col_id, first {}): {}\n", print_nnz, col_id_slice);
    fmt::print("Values (val, first {}): {}\n", print_nnz, val_slice);
    if (csr.nnz > kMaxPrint) {
      fmt::print("  ... ({} nonzeros omitted)\n", csr.nnz - kMaxPrint);
    }

    fmt::print("\nCOO Matrix test completed successfully!\n");
  } catch (const std::exception& e) {
    fmt::print(stderr, "Error: {}\n", e.what());
    return 1;
  }

  return 0;
}
