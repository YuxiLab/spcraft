#pragma once

#include <tuple>

#include "core/CscMatrix.h"

namespace spcraft
{

/**
 * @brief Doubly compressed sparse column storage, O(nnz + nzc), independent of n.
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
  /*************************************
   *             Data Members
   *************************************/
  OT* col_ptr = nullptr;  //!< offsets, size `nzc+1`
  IT* col_id = nullptr;   //!< logical column IDs, size nzc
  IT* row_id = nullptr;   //!< row IDs, size nnz
  NT* val = nullptr;      //!< values, size nnz
  OT nnz = 0;
  IT m = 0;
  IT n = 0;
  IT nzc = 0;
  bool memowned = true;

  /*************************************
   *             constructor
   *************************************/
  DcscMatrix() = default;
  // clang-format off
  //! non-owning view mode constructor
  DcscMatrix(OT* pointers, IT* columns, IT* rows, NT* values, OT nonzeros, IT nRows, IT nCols,IT storedColumns):col_ptr(pointers),col_id(columns),row_id(rows),val(values),nnz(nonzeros),m(nRows),n(nCols),nzc(storedColumns),memowned(false){}
  // clang-format on
  //! disable copy constructor
  DcscMatrix(const DcscMatrix&) = delete;
  //! disable copy operator
  DcscMatrix& operator=(const DcscMatrix&) = delete;
  // clang-format off
  //! move constructor
  DcscMatrix(DcscMatrix&& rhs) noexcept :col_ptr(rhs.col_ptr),col_id(rhs.col_id),row_id(rhs.row_id),val(rhs.val),nnz(rhs.nnz),m(rhs.m),n(rhs.n),nzc(rhs.nzc),memowned(rhs.memowned){rhs.Reset();}
  // clang-format on
  //! move operator
  DcscMatrix& operator=(DcscMatrix&& rhs) noexcept;
  //! deconstructor
  ~DcscMatrix() { SafeDelete(memowned, col_ptr, col_id, row_id, val); }

  //! Allocate zeroed replacement storage, preserving this object on failure.
  void Allocate(OT nonzeros, IT nRows, IT nCols, IT storedColumns);
  [[nodiscard]] DcscMatrix Clone() const;
  //! Release ownership without freeing. Use on moved-from objects or views.
  void Reset();
  //! Copy a well-formed CSC matrix, dropping empty column headers only.
  [[nodiscard]] static DcscMatrix FromCsc(const CscMatrix<IT, NT, OT>& source);
  //! Copy to CSC; this explicitly allocates n+1 offsets.
  [[nodiscard]] CscMatrix<IT, NT, OT> ToCsc() const;

 private:
  [[nodiscard]] static std::tuple<OT*, IT*, IT*, NT*> SafeAllocate(IT nzc, OT nnz);
  static void SafeDelete(bool owned, OT* pointers, IT* columns, IT* rows, NT* values);
};

}  // namespace spcraft

#include "core/DcscMatrix-inl.h"
