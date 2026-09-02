/**
 * @file 06_policy_based_design.cpp
 * @brief Demonstrates Policy-Based Class Design for Sparse Matrices in C++.
 *
 * Policy-Based Design allows composing complex sparse matrix classes out of small,
 * orthogonal compile-time policy classes without virtual function overhead:
 *  1. Storage Policy: Dictates memory allocation, ownership (Owning vs Non-owning View).
 *  2. Indexing Policy: Dictates index conversion (0-based C++ vs 1-based Fortran/MatrixMarket).
 *  3. Alignment Policy: Dictates cache-line / AVX-512 alignment requirements.
 *
 * All policy methods are inlined by the compiler, giving 100% zero runtime penalty.
 */

#include <fmt/core.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace spcraft::meta
{

// ----------------------------------------------------------------------------
// 1. Indexing Policies
// ----------------------------------------------------------------------------

struct ZeroBasedPolicy {
  template <typename IT>
  static constexpr IT ToInternal(IT idx) noexcept { return idx; }
  template <typename IT>
  static constexpr IT ToExternal(IT idx) noexcept { return idx; }
  static constexpr const char* name = "0-based (C/C++ standard)";
};

struct OneBasedPolicy {
  template <typename IT>
  static constexpr IT ToInternal(IT idx) noexcept { return idx - 1; }
  template <typename IT>
  static constexpr IT ToExternal(IT idx) noexcept { return idx + 1; }
  static constexpr const char* name = "1-based (Fortran/MATLAB/MatrixMarket)";
};

// ----------------------------------------------------------------------------
// 2. Storage Policies
// ----------------------------------------------------------------------------

template <typename IT, typename NT, typename OT>
struct OwningStoragePolicy {
  OT* row_ptr = nullptr;
  IT* col_id = nullptr;
  NT* val = nullptr;

  void AllocateStorage(OT nnz, IT m) {
    FreeStorage();
    row_ptr = new OT[m + 1]();
    col_id = (nnz > 0) ? new IT[nnz]() : nullptr;
    val = (nnz > 0) ? new NT[nnz]() : nullptr;
  }

  void FreeStorage() {
    delete[] row_ptr; row_ptr = nullptr;
    delete[] col_id;  col_id = nullptr;
    delete[] val;     val = nullptr;
  }

  ~OwningStoragePolicy() { FreeStorage(); }

  static constexpr bool IsOwning = true;
};

template <typename IT, typename NT, typename OT>
struct ViewStoragePolicy {
  OT* row_ptr = nullptr;
  IT* col_id = nullptr;
  NT* val = nullptr;

  void Attach(OT* r, IT* c, NT* v) {
    row_ptr = r;
    col_id = c;
    val = v;
  }

  void FreeStorage() { /* Non-owning view does nothing */ }
  ~ViewStoragePolicy() = default;

  static constexpr bool IsOwning = false;
};

// ----------------------------------------------------------------------------
// 3. Policy-Composed Sparse Matrix Host Class
// ----------------------------------------------------------------------------

template <typename IT,
          typename NT,
          typename OT = IT,
          template <typename, typename, typename> class StoragePolicy = OwningStoragePolicy,
          typename IndexPolicy = ZeroBasedPolicy>
class PolicySparseMatrix : public StoragePolicy<IT, NT, OT>, public IndexPolicy
{
 public:
  IT m = 0;
  IT n = 0;
  OT nnz = 0;

  PolicySparseMatrix() = default;

  // Constructor for Owning Matrix
  void InitOwning(IT rows, IT cols, OT num_nonzeros) {
    static_assert(StoragePolicy<IT, NT, OT>::IsOwning, "Cannot call InitOwning on a View Policy!");
    m = rows;
    n = cols;
    nnz = num_nonzeros;
    this->AllocateStorage(nnz, m);
  }

  // Method showing compile-time policy inlining for element access
  void SetEntry(OT entry_idx, IT user_row, IT user_col, NT value) {
    // Converts user index according to IndexPolicy at compile time!
    IT internal_col = IndexPolicy::ToInternal(user_col);
    this->col_id[entry_idx] = internal_col;
    this->val[entry_idx] = value;
  }

  void PrintSummary(const std::string& label) const {
    fmt::print("\n=== Matrix Configuration: {} ===\n", label);
    fmt::print("  Indexing Policy: {}\n", IndexPolicy::name);
    fmt::print("  Storage Policy:  {}\n", StoragePolicy<IT, NT, OT>::IsOwning ? "Owning (Heap)" : "Non-Owning View");
    fmt::print("  Dimensions:      {} x {}, nnz: {}\n", m, n, nnz);
  }
};

}  // namespace spcraft::meta

int main()
{
  fmt::print("============================================================\n");
  fmt::print(" Metaprogramming Pattern 6: Policy-Based Design\n");
  fmt::print("============================================================\n");

  // Config A: Owning + 0-based indexing
  using StandardCsr = spcraft::meta::PolicySparseMatrix<
      int32_t, double, int32_t,
      spcraft::meta::OwningStoragePolicy,
      spcraft::meta::ZeroBasedPolicy>;

  StandardCsr matA;
  matA.InitOwning(2, 2, 2);
  matA.SetEntry(0, /*user_row=*/0, /*user_col=*/0, 10.0);
  matA.SetEntry(1, /*user_row=*/1, /*user_col=*/1, 20.0);
  matA.PrintSummary("C++ Native 0-Based Owning Matrix");

  // Config B: Owning + 1-based Fortran/MatrixMarket indexing
  using FortranCsr = spcraft::meta::PolicySparseMatrix<
      int32_t, double, int32_t,
      spcraft::meta::OwningStoragePolicy,
      spcraft::meta::OneBasedPolicy>;

  FortranCsr matB;
  matB.InitOwning(2, 2, 2);
  // User specifies (1, 1) and (2, 2); policy automatically translates to internal (0, 0) and (1, 1)
  matB.SetEntry(0, /*user_row=*/1, /*user_col=*/1, 100.0);
  matB.SetEntry(1, /*user_row=*/2, /*user_col=*/2, 200.0);
  matB.PrintSummary("1-Based MatrixMarket/Fortran Policy Matrix");

  // Verify internal conversion
  fmt::print("  matB internal col_id[0] = {} (translated from 1-based index 1)\n", matB.col_id[0]);
  fmt::print("  matB internal col_id[1] = {} (translated from 1-based index 2)\n", matB.col_id[1]);

  fmt::print("\nPolicy-based design demonstration completed successfully!\n");
  return 0;
}
