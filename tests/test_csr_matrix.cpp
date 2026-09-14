// Ownership and lifetime contract for CsrMatrix. CsrMatrix is the reference
// implementation of the pattern the other containers follow, so this file also
// documents what test_coo_matrix and test_csc_matrix expect of theirs.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "SpCraft.h"

namespace
{

using Matrix = spcraft::CsrMatrix<std::int32_t, double, std::int64_t>;

int failures = 0;

void Check(bool condition, const char* what)
{
  if (!condition) {
    std::cerr << "CsrMatrix: " << what << "\n";
    ++failures;
  }
}

//! [ 1 0 2 ]
//! [ 0 3 0 ]
Matrix Sample()
{
  Matrix matrix;
  matrix.Allocate(3, 2, 3);
  matrix.row_ptr[0] = 0;
  matrix.row_ptr[1] = 2;
  matrix.row_ptr[2] = 3;
  matrix.col_id[0] = 0;
  matrix.col_id[1] = 2;
  matrix.col_id[2] = 1;
  matrix.val[0] = 1.0;
  matrix.val[1] = 2.0;
  matrix.val[2] = 3.0;
  return matrix;
}

}  // namespace

int main()
{
  // --- default construction owns nothing ---------------------------------
  {
    Matrix matrix;
    Check(matrix.row_ptr == nullptr && matrix.col_id == nullptr && matrix.val == nullptr,
          "a default-constructed matrix should hold no buffers");
    Check(matrix.nnz == 0 && matrix.m == 0 && matrix.n == 0,
          "a default-constructed matrix should be empty");
  }

  // --- Allocate zeroes the storage and records the shape -----------------
  {
    Matrix matrix;
    matrix.Allocate(4, 3, 5);
    Check(matrix.nnz == 4 && matrix.m == 3 && matrix.n == 5,
          "Allocate should record the shape");
    Check(matrix.memowned, "Allocate should take ownership");
    Check(std::all_of(matrix.row_ptr, matrix.row_ptr + 4, [](auto v) { return v == 0; }),
          "Allocate should zero row_ptr");
    Check(std::all_of(matrix.val, matrix.val + 4, [](auto v) { return v == 0.0; }),
          "Allocate should zero the values");
  }

  // --- an empty matrix is representable ----------------------------------
  {
    Matrix matrix;
    matrix.Allocate(0, 3, 3);
    Check(matrix.nnz == 0, "an empty matrix should allocate");
    Check(matrix.row_ptr != nullptr, "row_ptr is m+1 long even when nnz is zero");
    Matrix copy = matrix.Clone();
    Check(copy.nnz == 0 && copy.m == 3, "cloning an empty matrix should work");
  }

  // --- Allocate replaces previous storage without leaking ----------------
  {
    Matrix matrix;
    matrix.Allocate(4, 3, 3);
    matrix.Allocate(7, 5, 5);
    Check(matrix.nnz == 7 && matrix.m == 5, "a second Allocate should replace the first");
    Check(matrix.memowned, "a reallocated matrix still owns its storage");
  }

  // --- the pointer constructor is the borrow path ------------------------
  {
    std::int64_t row_ptr[3] = {0, 2, 3};
    std::int32_t col_id[3] = {0, 2, 1};
    double values[3] = {1.0, 2.0, 3.0};
    {
      Matrix view(row_ptr, col_id, values, 3, 2, 3);
      Check(!view.memowned, "the pointer constructor must not take ownership");
      Check(view.row_ptr == row_ptr, "a view should alias the caller's buffer");
    }
    // Reaching here without a crash means the destructor did not free the stack.
    Check(row_ptr[1] == 2, "a destroyed view must leave the caller's buffer alone");
  }

  // --- move construction transfers, and empties the source ---------------
  {
    Matrix source = Sample();
    const auto* buffer = source.val;
    Matrix moved(std::move(source));
    Check(moved.val == buffer, "move construction should steal the buffer");
    Check(moved.nnz == 3 && moved.m == 2 && moved.memowned, "move should carry the state");
    Check(source.val == nullptr && source.nnz == 0 && !source.memowned,
          "a moved-from matrix must release ownership");
  }

  // --- move assignment frees the target's old storage first --------------
  {
    Matrix target;
    target.Allocate(10, 4, 4);
    Matrix source = Sample();
    const auto* buffer = source.val;
    target = std::move(source);
    Check(target.val == buffer && target.nnz == 3,
          "move assignment should take the new storage");
    Check(source.val == nullptr && !source.memowned,
          "move assignment should empty the source");
  }

  // --- self-move assignment must not destroy the object ------------------
  {
    Matrix matrix = Sample();
    Matrix& alias = matrix;
    matrix = std::move(alias);
    Check(matrix.nnz == 3 && matrix.val != nullptr, "self-move must leave the matrix intact");
    Check(matrix.val[0] == 1.0, "self-move must not corrupt the values");
  }

  // --- Clone is a deep copy ----------------------------------------------
  {
    Matrix original = Sample();
    Matrix copy = original.Clone();
    Check(copy.val != original.val, "Clone must not alias the source");
    Check(copy.nnz == original.nnz && copy.m == original.m && copy.n == original.n,
          "Clone should preserve the shape");
    Check(std::equal(original.val, original.val + 3, copy.val),
          "Clone should copy the values");
    Check(std::equal(original.col_id, original.col_id + 3, copy.col_id),
          "Clone should copy the column indices");
    copy.val[0] = 99.0;
    Check(original.val[0] == 1.0, "writing to a clone must not touch the original");
  }

  // --- Reset releases without freeing ------------------------------------
  {
    Matrix matrix = Sample();
    auto* row_ptr = matrix.row_ptr;
    auto* col_id = matrix.col_id;
    auto* values = matrix.val;
    matrix.Reset();
    Check(matrix.row_ptr == nullptr && matrix.val == nullptr && matrix.nnz == 0 &&
              !matrix.memowned,
          "Reset should null every member and drop ownership");
    // Reset releases ownership without freeing, so the buffers are ours now.
    std::free(row_ptr);
    std::free(col_id);
    std::free(values);
  }

  // --- a request that cannot be sized is refused, not wrapped ------------
  {
    // nnz * sizeof(value) overflows std::size_t, so the multiplication would
    // wrap and hand back a buffer far smaller than requested.
    Matrix matrix;
    bool threw = false;
    try {
      matrix.Allocate(std::numeric_limits<std::int64_t>::max(), 10, 10);
    } catch (const std::bad_array_new_length&) {
      threw = true;
    }
    Check(threw, "an unsizeable allocation must throw std::bad_array_new_length");
    Check(matrix.nnz == 0 && matrix.val == nullptr,
          "a failed Allocate must leave the matrix untouched");
  }

  // --- negative sizes are rejected ---------------------------------------
  {
    Matrix matrix;
    bool threw = false;
    try {
      matrix.Allocate(-1, 2, 2);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "a negative nnz must be rejected");
  }

  if (failures != 0) {
    std::cerr << failures << " CsrMatrix check(s) failed\n";
    return 1;
  }
  std::cout << "CsrMatrix checks passed\n";
  return 0;
}
