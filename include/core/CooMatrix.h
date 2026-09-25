#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <fast_matrix_market/fast_matrix_market.hpp>
#include "core/CscMatrix.h"
#include "utils/omp/omp_wrapper.h"
#include "core/CsrMatrix.h"

namespace spcraft
{
template <class IT, class NT>
struct TupleEntry {
  IT row;
  IT col;
  NT val;
};

/**
 * @brief Coordinate (COO) storage with the native CombBLAS array-of-tuples layout.
 * Each Entry is (row, column, value); general COO has no ordering requirement.
 * SpGEMM returns entries sorted by column, then row, retaining structural zeros.
 * malloc/free own the storage; explicit construction starts std::tuple lifetimes.
 */
template <class IT, class NT, class OT = IT>
class CooMatrix
{
 public:
  /*************************************
   *             Data Members
   *************************************/
  using Entry = TupleEntry<IT, NT>;
  Entry* entries = nullptr;
  OT nnz = 0;
  IT m = 0;
  IT n = 0;
  bool memowned = true;
  /*************************************
   *             constructor
   *************************************/
  CooMatrix() = default;
  CooMatrix(TupleEntry<IT, NT>* entries_, OT count, IT rows, IT columns)
      : entries(entries_), nnz(count), m(rows), n(columns), memowned(false)
  {
  }
  CooMatrix(const CooMatrix&) = delete;
  CooMatrix& operator=(const CooMatrix&) = delete;
  CooMatrix(CooMatrix&& rhs) noexcept
      : entries(rhs.entries), nnz(rhs.nnz), m(rhs.m), n(rhs.n), memowned(rhs.memowned)
  {
    rhs.Reset();
  }
  CooMatrix& operator=(CooMatrix&& rhs) noexcept;
  ~CooMatrix() { SafeDelete(memowned, entries); }
  explicit CooMatrix(const std::string& filename);
  /*************************************
   *             member functions
   *************************************/
  void Allocate(OT count, IT rows, IT columns);
  [[nodiscard]] CooMatrix Clone() const;
  void Reset() noexcept;
  [[nodiscard]] CscMatrix<IT, NT, OT> ToCsc() const;
  [[nodiscard]] CsrMatrix<IT, NT, OT> ToCsr() const;

 private:
  [[nodiscard]] static Entry* SafeAllocate(OT count, IT rows, IT columns);
  static void SafeDelete(bool owned, Entry* entries) noexcept;
};
}  // namespace spcraft

#include "core/CooMatrix-inl.h"
