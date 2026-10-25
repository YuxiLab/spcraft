#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

using Matrix = DenseMatrix<int32_t, double>;

}  // namespace

TEST(DenseMatrix, RectangularColumnMajorAccess)
{
  // 1. Distinct row/column values detect an accidental row-major interpretation.
  Matrix matrix(2, 3);
  for (int32_t col = 0; col < matrix.n; ++col) {
    for (int32_t row = 0; row < matrix.m; ++row) {
      matrix[row, col] = 10 * col + row;
    }
  }
  EXPECT_EQ(matrix.m, 2);
  EXPECT_EQ(matrix.n, 3);
  EXPECT_EQ(matrix.ld, 2);
  EXPECT_EQ(matrix.size(), 6);
  const double expected[] = {0, 1, 10, 11, 20, 21};
  for (int32_t p = 0; p < 6; ++p) EXPECT_EQ(matrix.val[p], expected[p]);
  const auto& read_only = matrix;
  EXPECT_EQ((read_only[1, 2]), 21);
  EXPECT_EQ(read_only(1, 2), 21);
}

TEST(DenseMatrix, PaddedViewAndPackedClone)
{
  // 1. The final column needs no trailing padding; access must honor ld=4.
  double buffer[] = {1, 2, -99, -99, 3, 4};
  Matrix view(buffer, 2, 2, 4);
  EXPECT_FALSE(view.memowned);
  EXPECT_EQ(view.size(), 4);
  EXPECT_EQ((view[0, 1]), 3);
  view[1, 1] = 7;
  EXPECT_EQ(buffer[5], 7);
  EXPECT_EQ(buffer[2], -99);
  // 2. Cloning copies logical entries into independent packed storage.
  auto copy = view.Clone();
  EXPECT_TRUE(copy.memowned);
  EXPECT_NE(copy.val, buffer);
  EXPECT_EQ(copy.ld, 2);
  EXPECT_EQ(copy(0, 1), 3);
  EXPECT_EQ(copy(1, 1), 7);
  copy(0, 0) = 20;
  EXPECT_EQ(view(0, 0), 1);
  // 3. Clearing a view neither frees nor overwrites its external storage.
  view.Clear();
  EXPECT_EQ(view.val, nullptr);
  EXPECT_EQ(view.m, 0);
  EXPECT_EQ(view.n, 0);
  EXPECT_EQ(view.ld, 0);
  EXPECT_TRUE(view.memowned);
  EXPECT_EQ(buffer[0], 1);
  EXPECT_EQ(buffer[5], 7);
}

TEST(DenseMatrix, AllocateDetachesBorrowedStorage)
{
  // 1. Allocating a view produces an owner and leaves its old stack buffer alone.
  double buffer[] = {1, 2, 3, 4};
  Matrix view(buffer, 2, 2);
  view.Allocate(3, 2);
  EXPECT_TRUE(view.memowned);
  EXPECT_NE(view.val, buffer);
  EXPECT_EQ(view.m, 3);
  EXPECT_EQ(view.n, 2);
  EXPECT_EQ(view.ld, 3);
  EXPECT_EQ(buffer[3], 4);
}

TEST(DenseMatrix, ZeroAreaShapesArePreserved)
{
  // 1. Zero rows and zero columns are distinct valid shapes, including in clones.
  for (int32_t rows : {0, 3}) {
    for (int32_t cols : {0, 4}) {
      if (rows != 0 && cols != 0) continue;
      Matrix matrix(rows, cols);
      EXPECT_EQ(matrix.val, nullptr);
      EXPECT_EQ(matrix.size(), 0);
      EXPECT_EQ(matrix.m, rows);
      EXPECT_EQ(matrix.n, cols);
      auto copy = matrix.Clone();
      EXPECT_EQ(copy.m, rows);
      EXPECT_EQ(copy.n, cols);
      EXPECT_EQ(copy.ld, rows);
      EXPECT_EQ(copy.val, nullptr);
    }
  }
  Matrix empty_view(nullptr, 0, 4);
  EXPECT_FALSE(empty_view.memowned);
  EXPECT_EQ(empty_view.size(), 0);
}

TEST(DenseMatrix, RejectsMalformedViewsAndNegativeShapes)
{
  // 1. Reject malformed metadata before touching a caller's buffer.
  double buffer[4] = {};
  EXPECT_THROW(Matrix(-1, 2), std::invalid_argument);
  EXPECT_THROW(Matrix(2, -1), std::invalid_argument);
  EXPECT_THROW(Matrix(nullptr, 2, 2), std::invalid_argument);
  EXPECT_THROW(Matrix(buffer, 2, 2, 1), std::invalid_argument);
  EXPECT_THROW(Matrix(buffer, 0, 2, -1), std::invalid_argument);
}

TEST(DenseMatrix, AllocationOverflowPreservesExistingOwner)
{
  // 1. Reject a product that exceeds 32-bit offsets before calling malloc.
  Matrix matrix(2, 2, 7.0);
  auto* original = matrix.val;
  constexpr int32_t overflowing_square = 46341;  // Its square exceeds INT32_MAX.
  EXPECT_THROW(matrix.Allocate(overflowing_square, overflowing_square), std::bad_array_new_length);
  EXPECT_THROW(matrix.Allocate(-1, 2), std::invalid_argument);
  EXPECT_EQ(matrix.val, original);
  EXPECT_EQ(matrix.m, 2);
  EXPECT_EQ(matrix.n, 2);
  EXPECT_EQ(matrix(1, 1), 7);
  // 2. Independently check allocation bytes and a padded view's offset span.
  using WideMatrix = DenseMatrix<int64_t, double>;
  EXPECT_THROW(WideMatrix(std::numeric_limits<int64_t>::max(), 1), std::bad_array_new_length);
  using WideOffsetMatrix = DenseMatrix<int32_t, double, int64_t>;
  EXPECT_THROW(WideOffsetMatrix(original, 1, 3, std::numeric_limits<int64_t>::max()),
               std::bad_array_new_length);
}

template <class MatrixType>
class DenseMatrixTypeTest : public ::testing::Test
{
};

using DenseMatrixTypes =
    ::testing::Types<DenseMatrix<int32_t, float>, DenseMatrix<int32_t, double>, DenseMatrix<int32_t, bool>,
                     DenseMatrix<int32_t, float, int64_t>, DenseMatrix<int32_t, double, int64_t>,
                     DenseMatrix<int32_t, bool, int64_t>, DenseMatrix<int64_t, float>,
                     DenseMatrix<int64_t, double>, DenseMatrix<int64_t, bool>>;
TYPED_TEST_SUITE(DenseMatrixTypeTest, DenseMatrixTypes);

TYPED_TEST(DenseMatrixTypeTest, CompiledOwnershipMovesAndDeepClone)
{
  using Value = typename TypeParam::value_type;
  static_assert(!std::is_copy_constructible_v<TypeParam>);
  static_assert(!std::is_copy_assignable_v<TypeParam>);
  static_assert(std::is_nothrow_move_constructible_v<TypeParam>);
  static_assert(std::is_nothrow_move_assignable_v<TypeParam>);
  // 1. Check the filled constructor and an independent clone for every exported type.
  TypeParam source(2, 3, Value{1});
  auto clone = source.Clone();
  EXPECT_NE(clone.val, source.val);
  EXPECT_EQ(clone(1, 2), Value{1});
  clone[1, 2] = Value{0};
  const auto& read_only = source;
  EXPECT_EQ((read_only[1, 2]), Value{1});
  // 2. Transfer an owner twice and preserve it on self-move.
  auto* data = clone.val;
  TypeParam moved(std::move(clone));
  EXPECT_EQ(moved.val, data);
  EXPECT_EQ(clone.val, nullptr);
  EXPECT_EQ(clone.size(), 0);
  TypeParam target(1, 1);
  target = std::move(moved);
  EXPECT_EQ(target.val, data);
  EXPECT_EQ(moved.val, nullptr);
  auto& alias = target;
  target = std::move(alias);
  EXPECT_EQ(target.val, data);
  EXPECT_EQ(target(1, 2), Value{0});
  // 3. A moved view remains borrowed and can be cleared without freeing its buffer.
  Value buffer[] = {Value{0}, Value{1}};
  TypeParam view(buffer, 2, 1);
  TypeParam borrowed(std::move(view));
  EXPECT_FALSE(borrowed.memowned);
  EXPECT_EQ(borrowed.val, buffer);
  borrowed.Clear();
  EXPECT_EQ(buffer[1], Value{1});
  target.Clear();
  EXPECT_EQ(target.val, nullptr);
  EXPECT_EQ(target.m, 0);
  EXPECT_EQ(target.n, 0);
  EXPECT_EQ(target.ld, 0);
}

}  // namespace spcraft::tests
