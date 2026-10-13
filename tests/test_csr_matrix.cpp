#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <tuple>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

template <class Types>
class CsrMatrixTest : public ::testing::Test
{
 public:
  using IT = std::tuple_element_t<0, Types>;
  using NT = std::tuple_element_t<1, Types>;
  using OT = std::tuple_element_t<2, Types>;
  using Matrix = spcraft::CsrMatrix<IT, NT, OT>;
};
using CsrTypes = ::testing::Types<std::tuple<std::int32_t, float, std::int64_t>,
                                  std::tuple<std::int32_t, double, std::int64_t>,
                                  std::tuple<std::int64_t, double, std::int64_t>>;
TYPED_TEST_SUITE(CsrMatrixTest, CsrTypes);

//! [ 1 0 2 ]
//! [ 0 3 0 ]
template <class Matrix>
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

TYPED_TEST(CsrMatrixTest, DefaultConstructionOwnsNothing)
{
  using Matrix = typename TestFixture::Matrix;
  // --- default construction owns nothing ---------------------------------

  Matrix matrix;
  EXPECT_TRUE((matrix.row_ptr == nullptr && matrix.col_id == nullptr && matrix.val == nullptr))
      << "a default-constructed matrix should hold no buffers";
  EXPECT_TRUE((matrix.nnz == 0 && matrix.m == 0 && matrix.n == 0))
      << "a default-constructed matrix should be empty";
}

TYPED_TEST(CsrMatrixTest, AllocateRecordsShapeAndOwnsStorage)
{
  using Matrix = typename TestFixture::Matrix;

  // Allocate reserves storage; callers initialize its contents.

  Matrix matrix;
  matrix.Allocate(4, 3, 5);
  EXPECT_TRUE((matrix.nnz == 4 && matrix.m == 3 && matrix.n == 5))
      << "Allocate should record the shape";
  EXPECT_TRUE((matrix.memowned)) << "Allocate should take ownership";
  EXPECT_NE(matrix.row_ptr, nullptr);
  EXPECT_NE(matrix.col_id, nullptr);
  EXPECT_NE(matrix.val, nullptr);
}

TYPED_TEST(CsrMatrixTest, AnEmptyMatrixIsRepresentable)
{
  using Matrix = typename TestFixture::Matrix;
  using OT = typename TestFixture::OT;

  // --- an empty matrix is representable ----------------------------------

  Matrix matrix;
  matrix.Allocate(0, 3, 3);
  std::fill(matrix.row_ptr, matrix.row_ptr + matrix.m + 1, OT{0});
  EXPECT_TRUE((matrix.nnz == 0)) << "an empty matrix should allocate";
  EXPECT_TRUE((matrix.row_ptr != nullptr)) << "row_ptr is m+1 long even when nnz is zero";
  Matrix copy = matrix.Clone();
  EXPECT_TRUE((copy.nnz == 0 && copy.m == 3)) << "cloning an empty matrix should work";
}

TYPED_TEST(CsrMatrixTest, AllocateReplacesPreviousStorageWithoutLeaking)
{
  using Matrix = typename TestFixture::Matrix;

  // --- Allocate replaces previous storage without leaking ----------------

  Matrix matrix;
  matrix.Allocate(4, 3, 3);
  matrix.Allocate(7, 5, 5);
  EXPECT_TRUE((matrix.nnz == 7 && matrix.m == 5)) << "a second Allocate should replace the first";
  EXPECT_TRUE((matrix.memowned)) << "a reallocated matrix still owns its storage";
}

TYPED_TEST(CsrMatrixTest, ThePointerConstructorIsTheBorrowPath)
{
  using Matrix = typename TestFixture::Matrix;
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;

  // --- the pointer constructor is the borrow path ------------------------

  OT row_ptr[3] = {0, 2, 3};
  IT col_id[3] = {0, 2, 1};
  NT values[3] = {1.0, 2.0, 3.0};
  {
    Matrix view(row_ptr, col_id, values, 3, 2, 3);
    EXPECT_TRUE((!view.memowned)) << "the pointer constructor must not take ownership";
    EXPECT_TRUE((view.row_ptr == row_ptr)) << "a view should alias the caller's buffer";
  }
  // Reaching here without a crash means the destructor did not free the stack.
  EXPECT_TRUE((row_ptr[1] == 2)) << "a destroyed view must leave the caller's buffer alone";
}

TYPED_TEST(CsrMatrixTest, MoveConstructionTransfersAndEmptiesTheSource)
{
  using Matrix = typename TestFixture::Matrix;

  // --- move construction transfers, and empties the source ---------------

  Matrix source = Sample<Matrix>();
  const auto* buffer = source.val;
  Matrix moved(std::move(source));
  EXPECT_TRUE((moved.val == buffer)) << "move construction should steal the buffer";
  EXPECT_TRUE((moved.nnz == 3 && moved.m == 2 && moved.memowned)) << "move should carry the state";
  EXPECT_TRUE((source.val == nullptr && source.nnz == 0 && !source.memowned))
      << "a moved-from matrix must release ownership";
}

TYPED_TEST(CsrMatrixTest, MoveAssignmentFreesTheTargetSOldStorage)
{
  using Matrix = typename TestFixture::Matrix;

  // --- move assignment frees the target's old storage first --------------

  Matrix target;
  target.Allocate(10, 4, 4);
  Matrix source = Sample<Matrix>();
  const auto* buffer = source.val;
  target = std::move(source);
  EXPECT_TRUE((target.val == buffer && target.nnz == 3))
      << "move assignment should take the new storage";
  EXPECT_TRUE((source.val == nullptr && !source.memowned))
      << "move assignment should empty the source";
}

TYPED_TEST(CsrMatrixTest, SelfMoveAssignmentMustNotDestroyTheObject)
{
  using Matrix = typename TestFixture::Matrix;

  // --- self-move assignment must not destroy the object ------------------

  Matrix matrix = Sample<Matrix>();
  Matrix& alias = matrix;
  matrix = std::move(alias);
  EXPECT_TRUE((matrix.nnz == 3 && matrix.val != nullptr))
      << "self-move must leave the matrix intact";
  EXPECT_TRUE((matrix.val[0] == 1.0)) << "self-move must not corrupt the values";
}

TYPED_TEST(CsrMatrixTest, CloneIsADeepCopy)
{
  using Matrix = typename TestFixture::Matrix;

  // --- Clone is a deep copy ----------------------------------------------

  Matrix original = Sample<Matrix>();
  Matrix copy = original.Clone();
  EXPECT_TRUE((copy.val != original.val)) << "Clone must not alias the source";
  EXPECT_TRUE((copy.nnz == original.nnz && copy.m == original.m && copy.n == original.n))
      << "Clone should preserve the shape";
  EXPECT_TRUE((std::equal(original.val, original.val + 3, copy.val)))
      << "Clone should copy the values";
  EXPECT_TRUE((std::equal(original.col_id, original.col_id + 3, copy.col_id)))
      << "Clone should copy the column indices";
  copy.val[0] = 99.0;
  EXPECT_TRUE((original.val[0] == 1.0)) << "writing to a clone must not touch the original";
}

TYPED_TEST(CsrMatrixTest, ResetReleasesWithoutFreeing)
{
  using Matrix = typename TestFixture::Matrix;

  // --- Reset releases without freeing ------------------------------------

  Matrix matrix = Sample<Matrix>();
  auto* row_ptr = matrix.row_ptr;
  auto* col_id = matrix.col_id;
  auto* values = matrix.val;
  matrix.Reset();
  EXPECT_TRUE(
      (matrix.row_ptr == nullptr && matrix.val == nullptr && matrix.nnz == 0 && !matrix.memowned))
      << "Reset should null every member and drop ownership";
  // Reset releases ownership without freeing, so the buffers are ours now.
  std::free(row_ptr);
  std::free(col_id);
  std::free(values);
}

TYPED_TEST(CsrMatrixTest, ARequestThatCannotBeSizedIsRefused)
{
  using Matrix = typename TestFixture::Matrix;

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

TYPED_TEST(CsrMatrixTest, NegativeSizesAreRejected)
{
  using Matrix = typename TestFixture::Matrix;

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
