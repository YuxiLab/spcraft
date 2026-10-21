#pragma once

#include <cstdint>
#include <string>

#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{
template <class IT, class NT>
struct TupleEntry {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT row{};
  IT col{};
  NT val{};
};

/**
 * @brief Coordinate (COO) storage with the native CombBLAS array-of-tuples layout.
 * Link SPCraft for float, double, or bool values with (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64), using signed fixed-width integer types.
 * Each Entry is (row, column, value); general COO has no ordering requirement.
 * SpGEMM returns entries sorted by column, then row, retaining structural zeros.
 * malloc/free own the storage of trivial Entry values.
 */
template <class IT, class NT, class OT = IT>
class CooMatrix
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  using Entry = TupleEntry<IT, NT>;
  Entry* entries = nullptr;
  OT nnz = 0;
  IT m = 0;
  IT n = 0;
  bool memowned = true;

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  CooMatrix() noexcept;

  CooMatrix(TupleEntry<IT, NT>* entries_, OT count, IT rows, IT columns);

  CooMatrix(const CooMatrix&) = delete;

  CooMatrix& operator=(const CooMatrix&) = delete;

  CooMatrix(CooMatrix&& rhs) noexcept;

  CooMatrix& operator=(CooMatrix&& rhs) noexcept;

  ~CooMatrix();

  explicit CooMatrix(const std::string& filename);

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  void Allocate(OT count, IT rows, IT columns);

  [[nodiscard]] CooMatrix Clone() const;

  void Reset() noexcept;

  [[nodiscard]] SPCSC ToCsc() const;

  [[nodiscard]] SPCSR ToCsr() const;

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  [[nodiscard]] static Entry* SafeAllocate(OT count, IT rows, IT columns);

  static void SafeDelete(bool owned, Entry* entries) noexcept;
};
}  // namespace spcraft
