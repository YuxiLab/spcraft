#pragma once

#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Owning dense vector or non-owning view of an existing buffer.
 * Link SPCraft for float, double, or bool values with signed int32_t/int64_t indices.
 *
 * Memory allocated by this class is host memory. Use the pointer constructor
 * to wrap externally managed memory, including CUDA device memory.
 */
template <class IT, class NT>
class DenseVector
{
 public:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  NT* val = nullptr;     //!< Elements, size n.
  IT n = 0;              //!< Number of elements.
  bool memowned = true;  //!< Owns the storage (views opt out).

  /*****************************************************************************
   *                               Constructors                                *
   ****************************************************************************/
  DenseVector() noexcept;

  //! Allocate an uninitialized vector.
  explicit DenseVector(IT size);

  //! Allocate and initialize with `init` value.
  DenseVector(IT size, NT init);

  //! Wrap an externally managed buffer as a non-owning view.
  DenseVector(NT* val_, IT size);

  DenseVector(const DenseVector&) = delete;

  DenseVector(DenseVector&& rhs) noexcept;

  ~DenseVector();

  DenseVector& operator=(const DenseVector&) = delete;

  DenseVector& operator=(DenseVector&& rhs) noexcept;

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  SPND IT size() const noexcept { return n; }

  //! Unchecked element access; storage must be host-accessible.
  NT& operator[](IT index) noexcept { return val[index]; }

  const NT& operator[](IT index) const noexcept { return val[index]; }

  //! Allocate replacement storage before freeing the current owned buffer.
  void Allocate(IT size);

  //! Return an owning copy of host-accessible storage.
  SPND DenseVector Clone() const;

  //! Release ownership without freeing; calling this on live owned storage leaks it.
  void Reset() noexcept;

  //! Fill host-accessible storage using the given random seed.
  //! Reals use [-1, 1); bool values are false or true.
  void Random(int seed);

 private:
  /*****************************************************************************
   *                              Static Members                               *
   ****************************************************************************/
  SPND static NT* SafeAllocate(IT size);

  static void SafeDelete(bool owned, NT* ptr) noexcept;
};

}  // namespace spcraft
