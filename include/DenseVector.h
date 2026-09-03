#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
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

  [[nodiscard]] static NT* SafeAllocate(IT size)
  {
    if (size < IT{0}) {
      throw std::invalid_argument("DenseVector size must be non-negative");
    }
    if (size == IT{0}) return nullptr;
    return new NT[static_cast<std::size_t>(size)]{};
  }

  static void SafeDelete(bool memowned, NT* val)
  {
    if (memowned) delete[] val;
  }

 public:
  NT* val = nullptr;     //!< Elements, size n.
  IT n = 0;              //!< Number of elements.
  bool memowned = true;  //!< Owns the storage (views opt out).

  //! Empty vector.
  DenseVector() = default;

  //! Allocate a zero-initialized vector.
  explicit DenseVector(IT size)
  {
    Allocate(size);
  }

  //! Wrap an externally managed buffer as a non-owning view.
  DenseVector(NT* val_, IT size) : val(val_), n(size), memowned(false)
  {
    if (size < IT{0}) {
      throw std::invalid_argument("DenseVector size must be non-negative");
    }
    if (size != IT{0} && val == nullptr) {
      throw std::invalid_argument("DenseVector data must not be null for a non-empty vector");
    }
  }

  DenseVector(const DenseVector&) = delete;
  DenseVector& operator=(const DenseVector&) = delete;

  DenseVector(DenseVector&& rhs) noexcept : val(rhs.val), n(rhs.n), memowned(rhs.memowned)
  {
    rhs.Reset();
  }

  DenseVector& operator=(DenseVector&& rhs) noexcept
  {
    if (this != &rhs) {
      SafeDelete(memowned, val);
      val = rhs.val;
      n = rhs.n;
      memowned = rhs.memowned;
      rhs.Reset();
    }
    return *this;
  }

  ~DenseVector()
  {
    SafeDelete(memowned, val);
  }

  //! Replace the current storage with a zero-initialized owned buffer.
  void Allocate(IT size)
  {
    if (size < IT{0}) {
      throw std::invalid_argument("DenseVector size must be non-negative");
    }
    NT* replacement = SafeAllocate(size);
    SafeDelete(memowned, val);
    val = replacement;
    n = size;
    memowned = true;
  }

  //! Return an owning copy.
  [[nodiscard]] DenseVector Clone() const
  {
    DenseVector result(n);
    if (n != IT{0}) {
      std::copy(val, val + n, result.val);
    }
    return result;
  }

  //! Null every member without freeing; releases ownership.
  void Reset() noexcept
  {
    val = nullptr;
    n = 0;
    memowned = false;
  }

  [[nodiscard]] NT* data() noexcept
  {
    return val;
  }
  [[nodiscard]] const NT* data() const noexcept
  {
    return val;
  }
  [[nodiscard]] NT* begin() noexcept
  {
    return val;
  }
  [[nodiscard]] const NT* begin() const noexcept
  {
    return val;
  }
  [[nodiscard]] NT* end() noexcept
  {
    return n == IT{0} ? val : val + n;
  }
  [[nodiscard]] const NT* end() const noexcept
  {
    return n == IT{0} ? val : val + n;
  }

  NT& operator[](IT index) noexcept
  {
    return val[index];
  }
  const NT& operator[](IT index) const noexcept
  {
    return val[index];
  }
};

}  // namespace spcraft
