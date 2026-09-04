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
  static_assert(std::is_integral_v<IT>, "DenseVector index type must be integral");
  static_assert(!std::is_const_v<NT>, "DenseVector value type must not be const");

  [[nodiscard]] static NT* SafeAllocate(IT size);
  static void SafeDelete(bool memowned, NT* val);

 public:
  NT* val = nullptr;     //!< Elements, size n.
  IT n = 0;              //!< Number of elements.
  bool memowned = true;  //!< Owns the storage (views opt out).

  //! Empty vector.
  DenseVector() = default;

  //! Allocate an uninitialized vector.
  explicit DenseVector(IT size);

  //! Wrap an externally managed buffer as a non-owning view.
  DenseVector(NT* val_, IT size);
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
};

}  // namespace spcraft

#include "core/DenseVector-inl.h"
