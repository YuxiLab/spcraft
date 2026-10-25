#pragma once

namespace spcraft
{

// Column-major host matrix. Link SPCraft for float, double, or bool with signed
// (IT,OT) widths (32,32), (32,64), or (64,64). Copies are explicit through Clone.
template <class IT, class NT, class OT = IT>
class DenseMatrix
{
 public:
  using index_type = IT;
  using value_type = NT;
  using offset_type = OT;

  NT* val = nullptr;     // First matrix entry; views borrow their caller's buffer.
  IT m = 0;              // Number of rows.
  IT n = 0;              // Number of columns.
  OT ld = 0;             // Distance between column starts, in elements.
  bool memowned = true;  // Owns val; false for a view.

  DenseMatrix() noexcept;
  DenseMatrix(IT rows, IT cols);
  DenseMatrix(IT rows, IT cols, NT value);
  DenseMatrix(NT* values, IT rows, IT cols);
  DenseMatrix(NT* values, IT rows, IT cols, OT leading_dim);
  DenseMatrix(const DenseMatrix&) = delete;
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix(DenseMatrix&& other) noexcept;
  DenseMatrix& operator=(DenseMatrix&& other) noexcept;
  ~DenseMatrix();

  // Logical entry count; padded views can span more storage than this count.
  [[nodiscard]] OT size() const noexcept { return static_cast<OT>(m) * static_cast<OT>(n); }

  // Unchecked access; row/column bounds and host accessibility are caller contracts.
  NT& operator()(IT row, IT col) noexcept { return val[static_cast<OT>(row) + static_cast<OT>(col) * ld]; }
  const NT& operator()(IT row, IT col) const noexcept
  {
    return val[static_cast<OT>(row) + static_cast<OT>(col) * ld];
  }

#if defined(__cpp_multidimensional_subscript) && __cpp_multidimensional_subscript >= 202110L
  // C++23 element indexing; CUDA C++20 callers can use operator().
  NT& operator[](IT row, IT col) noexcept { return (*this)(row, col); }
  const NT& operator[](IT row, IT col) const noexcept { return (*this)(row, col); }
#endif

  // Allocate uninitialized packed storage (ld=rows); failure preserves this matrix.
  void Allocate(IT rows, IT cols);
  [[nodiscard]] DenseMatrix Clone() const;
  // Free owned storage, detach a view, and restore the default empty state.
  void Clear() noexcept;

 private:
  // Validate dimensions, addressable span, offset range, and allocation byte count.
  [[nodiscard]] static OT CheckedSpan(IT rows, IT cols, OT leading_dim);
};

}  // namespace spcraft
