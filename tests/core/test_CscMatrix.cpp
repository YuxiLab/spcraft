#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

using Matrix = spcraft::CscMatrix<std::int32_t, double, std::int64_t>;

//! [ 1 0 ]
//! [ 0 3 ]
//! [ 2 0 ]   stored by column: col 0 = rows {0, 2}, col 1 = row {1}
Matrix Sample()
{
  Matrix matrix;
  matrix.Allocate(3, 3, 2);
  matrix.col_ptr[0] = 0;
  matrix.col_ptr[1] = 2;
  matrix.col_ptr[2] = 3;
  matrix.row_id[0] = 0;
  matrix.row_id[1] = 2;
  matrix.row_id[2] = 1;
  matrix.val[0] = 1.0;
  matrix.val[1] = 2.0;
  matrix.val[2] = 3.0;
  return matrix;
}

}  // namespace

TEST(CscMatrix, DefaultConstructionOwnsNothing)
{
  // --- default construction owns nothing ---------------------------------

  Matrix matrix;
  EXPECT_TRUE((matrix.col_ptr == nullptr && matrix.row_id == nullptr && matrix.val == nullptr))
      << "a default-constructed matrix should hold no buffers";
  EXPECT_TRUE((matrix.nnz == 0 && matrix.m == 0 && matrix.n == 0))
      << "a default-constructed matrix should be empty";
}

TEST(CscMatrix, AllocateZeroesTheStorageAndRecordsTheShape)
{
  // --- Allocate zeroes the storage and records the shape -----------------

  Matrix matrix;
  matrix.Allocate(4, 5, 3);
  EXPECT_TRUE((matrix.nnz == 4 && matrix.m == 5 && matrix.n == 3)) << "Allocate should record the shape";
  EXPECT_TRUE((matrix.memowned)) << "Allocate should take ownership";
  EXPECT_TRUE((std::all_of(matrix.col_ptr, matrix.col_ptr + 4, [](auto v) { return v == 0; })))
      << "Allocate should zero col_ptr";
  EXPECT_TRUE((std::all_of(matrix.val, matrix.val + 4, [](auto v) { return v == 0.0; })))
      << "Allocate should zero the values";
}

TEST(CscMatrix, AnEmptyMatrixIsRepresentable)
{
  // --- an empty matrix is representable ----------------------------------

  Matrix matrix;
  bool threw = false;
  try {
    matrix.Allocate(0, 3, 3);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE((!threw)) << "allocating an empty matrix must be allowed";
  EXPECT_TRUE((matrix.nnz == 0 && matrix.n == 3)) << "an empty matrix should still carry its shape";
  EXPECT_TRUE((matrix.col_ptr != nullptr)) << "col_ptr is n+1 long even when nnz is zero";
}

TEST(CscMatrix, CloningAnEmptyMatrix)
{
  // --- cloning an empty matrix -------------------------------------------

  Matrix matrix;
  bool threw = false;
  try {
    Matrix copy = matrix.Clone();
    EXPECT_TRUE((copy.nnz == 0)) << "cloning an empty matrix should give an empty matrix";
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE((!threw)) << "cloning a default-constructed matrix must not throw";
}

TEST(CscMatrix, ThePointerConstructorIsTheBorrowPath)
{
  // --- the pointer constructor is the borrow path ------------------------

  std::int64_t col_ptr[3] = {0, 2, 3};
  std::int32_t row_id[3] = {0, 2, 1};
  double values[3] = {1.0, 2.0, 3.0};
  {
    Matrix view(col_ptr, row_id, values, 3, 3, 2);
    EXPECT_TRUE((!view.memowned)) << "the pointer constructor must not take ownership";
    EXPECT_TRUE((view.col_ptr == col_ptr)) << "a view should alias the caller's buffer";
  }
  EXPECT_TRUE((col_ptr[1] == 2)) << "a destroyed view must leave the caller's buffer alone";
}

TEST(CscMatrix, MoveConstructionAndAssignment)
{
  // --- move construction and assignment ----------------------------------

  Matrix source = Sample();
  const auto* buffer = source.val;
  Matrix moved(std::move(source));
  EXPECT_TRUE((moved.val == buffer && moved.nnz == 3)) << "move construction should steal the buffer";
  EXPECT_TRUE((source.val == nullptr && source.nnz == 0 && !source.memowned))
      << "a moved-from matrix must release ownership";

  Matrix target;
  target.Allocate(8, 4, 4);
  target = std::move(moved);
  EXPECT_TRUE((target.val == buffer && target.nnz == 3)) << "move assignment should take the new storage";
  EXPECT_TRUE((moved.val == nullptr && !moved.memowned)) << "move assignment should empty the source";
}

TEST(CscMatrix, SelfMoveAssignmentMustNotDestroyTheObject)
{
  // --- self-move assignment must not destroy the object ------------------

  Matrix matrix = Sample();
  Matrix& alias = matrix;
  matrix = std::move(alias);
  EXPECT_TRUE((matrix.nnz == 3 && matrix.val != nullptr)) << "self-move must leave the matrix intact";
}

TEST(CscMatrix, CloneIsADeepCopy)
{
  // --- Clone is a deep copy ----------------------------------------------

  Matrix original = Sample();
  Matrix copy = original.Clone();
  EXPECT_TRUE((copy.val != original.val)) << "Clone must not alias the source";
  EXPECT_TRUE((copy.nnz == original.nnz && copy.m == original.m && copy.n == original.n))
      << "Clone should preserve the shape";
  EXPECT_TRUE((std::equal(original.val, original.val + 3, copy.val))) << "Clone should copy the values";
  EXPECT_TRUE((std::equal(original.col_ptr, original.col_ptr + 3, copy.col_ptr)))
      << "Clone should copy the column pointers";
  copy.val[0] = 99.0;
  EXPECT_TRUE((original.val[0] == 1.0)) << "writing to a clone must not touch the original";
}

TEST(CscMatrix, ResetReleasesWithoutFreeing)
{
  // --- Reset releases without freeing ------------------------------------

  Matrix matrix = Sample();
  auto* col_ptr = matrix.col_ptr;
  auto* row_id = matrix.row_id;
  auto* values = matrix.val;
  matrix.Reset();
  EXPECT_TRUE((matrix.col_ptr == nullptr && matrix.nnz == 0 && !matrix.memowned))
      << "Reset should null every member and drop ownership";
  std::free(col_ptr);
  std::free(row_id);
  std::free(values);
}

TEST(CscMatrix, ARequestThatCannotBeSizedIsRefused)
{
  // --- a request that cannot be sized is refused, not wrapped ------------

  // nnz * sizeof(value) overflows std::size_t, so the multiplication would
  // wrap and hand back a buffer far smaller than requested.
  Matrix matrix;
  bool threw = false;
  try {
    matrix.Allocate(std::numeric_limits<std::int64_t>::max(), 10, 10);
  } catch (const std::bad_array_new_length&) {
    threw = true;
  }
  EXPECT_TRUE((threw)) << "an unsizeable allocation must throw std::bad_array_new_length";
  EXPECT_TRUE((matrix.nnz == 0 && matrix.val == nullptr))
      << "a failed Allocate must leave the matrix untouched";
}

TEST(CscMatrix, NegativeSizesAreRejected)
{
  // --- negative sizes are rejected ---------------------------------------

  Matrix matrix;
  bool threw = false;
  try {
    matrix.Allocate(-1, 2, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_TRUE((threw)) << "a negative nnz must be rejected";
}

}  // namespace spcraft::tests
