#pragma once

#include <type_traits>

namespace spcraft
{

/**
 * @brief Owning dense vector or non-owning view of an existing buffer.
 *
 * Memory allocated by this class is host memory. Use the pointer constructor
 * to wrap externally managed memory, including CUDA device memory.
 */
template <class IT, class NT>
class DenseVector
{
 public:
  NT* val = nullptr;     //!< Elements, size n.
  IT n = 0;              //!< Number of elements.
  bool memowned = true;  //!< Owns the storage (views opt out).

  DenseVector() = default;
  //! Allocate an uninitialized vector.
  explicit DenseVector(IT size);
  //! Allocate and initialize with `init` value.
  DenseVector(IT size, NT init);
  //! Wrap an externally managed buffer as a non-owning view.
  DenseVector(NT* val_, IT size) : val(val_), n(size), memowned(false) {}
  DenseVector(const DenseVector&) = delete;
  DenseVector& operator=(const DenseVector&) = delete;

  DenseVector(DenseVector&& rhs) noexcept;
  DenseVector& operator=(DenseVector&& rhs) noexcept;
  ~DenseVector();

  //! Replace the current storage with an uninitialized owned buffer.
  void Allocate(IT size);
  //! Return an owning copy.
  [[nodiscard]] DenseVector Clone() const;
  //! Null every member without freeing; releases ownership.
  void Reset() noexcept;

  //! Fill host-accessible storage using the given random seed.
  //! Reals use [-1, 1); integers use [-1, 1], or [0, 1] for unsigned/bool.
  void Random(int seed);

 private:
  [[nodiscard]] static NT* SafeAllocate(IT size);
  static void SafeDelete(bool owned, NT* ptr) noexcept;
};

}  // namespace spcraft

#include "core/DenseVector-inl.h"
