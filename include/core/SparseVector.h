#pragma once

#include <tuple>

#include "core/DenseVector.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Owning sparse vector or non-owning view of existing index/value arrays.
 * Link SPCraft for float, double, or bool values with signed int32_t/int64_t indices.
 *
 * Stores the nnz explicit entries of a length-n vector as parallel arrays:
 * `ind[k]` is the position of `val[k]`. Indices are strictly increasing and lie
 * in [0, n); kernels rely on that order, and ToDense() checks it. Allocate
 * leaves both arrays uninitialized, so the caller writes them in that order.
 * The entry count needs no separate offset type because nnz never exceeds n.
 *
 * Memory allocated by this class is host memory. Use the pointer constructor
 * to wrap externally managed memory, including CUDA device memory.
 */
template <class IT, class NT>
class SparseVector
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT* ind = nullptr;     //!< Positions of the stored entries, size nnz.
  NT* val = nullptr;     //!< Stored values, size nnz.
  IT nnz = 0;            //!< Number of stored entries.
  IT n = 0;              //!< Length of the vector.
  bool memowned = true;  //!< Owns the storage (views opt out).

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  SparseVector() noexcept;

  //! Allocate uninitialized storage for `count` entries of a length-`length` vector.
  SparseVector(IT count, IT length);

  //! Wrap externally managed index and value arrays as a non-owning view.
  SparseVector(IT* ind_, NT* val_, IT count, IT length);

  SparseVector(const SparseVector&) = delete;

  SparseVector(SparseVector&& rhs) noexcept;

  ~SparseVector();

  SparseVector& operator=(const SparseVector&) = delete;

  SparseVector& operator=(SparseVector&& rhs) noexcept;

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  [[nodiscard]] IT size() const noexcept { return n; }

  //! Allocate replacement storage before freeing the current owned arrays.
  void Allocate(IT count, IT length);

  //! Return an owning copy of host-accessible storage.
  [[nodiscard]] SparseVector Clone() const;

  //! Release ownership without freeing; calling this on live owned storage leaks it.
  void Reset() noexcept;

  //! Expand to a dense vector; positions without a stored entry hold `fill`.
  //! @throws std::invalid_argument when indices are out of range or not strictly increasing.
  [[nodiscard]] SPDVEC ToDense(NT fill) const;

  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  //! Keep every element of `dense` that differs from `fill`, in index order.
  [[nodiscard]] static SparseVector FromDense(const SPDVEC& dense, NT fill);

 private:
  [[nodiscard]] static std::tuple<IT*, NT*> SafeAllocate(IT count, IT length);

  static void SafeDelete(bool owned, IT* ind, NT* val) noexcept;
};

}  // namespace spcraft
