#pragma once

#include <tuple>

#include "core/CscMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Doubly compressed sparse column storage, O(nnz + nzc), independent of n.
 * Link SPCraft for float, double, or bool values with (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64), using signed fixed-width integer types.
 * # DCSC Format
 * col_id lists the nzc nonempty columns in strictly increasing order. col_ptr
 * indexes those stored columns, not the logical columns. Rows within a column
 * may be unsorted or repeated. A shaped empty matrix has nzc == nnz == 0.
 * Like the other host containers, buffers store implicit-lifetime value types.
 */
template <class IT, class NT, class OT = IT>
class DcscMatrix
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  OT* col_ptr = nullptr;  //!< offsets, size `nzc+1`
  IT* col_id = nullptr;   //!< logical column IDs, size nzc
  IT* row_id = nullptr;   //!< row IDs, size nnz
  NT* val = nullptr;      //!< values, size nnz
  OT nnz = 0;
  IT m = 0;
  IT n = 0;
  IT nzc = 0;
  bool memowned = true;

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  DcscMatrix() noexcept;

  //! non-owning view mode constructor
  DcscMatrix(OT* pointers, IT* columns, IT* rows, NT* values, OT nonzeros, IT nRows, IT nCols,
             IT storedColumns);

  //! disable copy constructor
  DcscMatrix(const DcscMatrix&) = delete;

  //! disable copy operator
  DcscMatrix& operator=(const DcscMatrix&) = delete;

  //! move constructor
  DcscMatrix(DcscMatrix&& rhs) noexcept;

  //! move operator
  DcscMatrix& operator=(DcscMatrix&& rhs) noexcept;

  //! deconstructor
  ~DcscMatrix();

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  //! Allocate zeroed replacement storage, preserving this object on failure.
  void Allocate(OT nonzeros, IT nRows, IT nCols, IT storedColumns);

  [[nodiscard]] DcscMatrix Clone() const;

  //! Release ownership without freeing. Use on moved-from objects or views.
  void Reset();

  //! Copy to CSC; this explicitly allocates n+1 offsets.
  [[nodiscard]] SPCSC ToCsc() const;

  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  //! Copy a well-formed CSC matrix, dropping empty column headers only.
  [[nodiscard]] static DcscMatrix FromCsc(const SPCSC& source);

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  [[nodiscard]] static std::tuple<OT*, IT*, IT*, NT*> SafeAllocate(IT nzc, OT nnz);

  static void SafeDelete(bool owned, OT* pointers, IT* columns, IT* rows, NT* values);
};

}  // namespace spcraft
