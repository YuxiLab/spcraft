#pragma once

#include "SpCraft.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace spcraft;

using DistributedTypes =
    testing::Types<std::tuple<int32_t, float, int32_t>, std::tuple<int32_t, double, int32_t>,
                   std::tuple<int32_t, bool, int32_t>, std::tuple<int32_t, float, int64_t>,
                   std::tuple<int32_t, double, int64_t>, std::tuple<int32_t, bool, int64_t>,
                   std::tuple<int64_t, float, int64_t>, std::tuple<int64_t, double, int64_t>,
                   std::tuple<int64_t, bool, int64_t>>;

template <template <class, class, class> class Distributed, class Types>
class DistributedMatrixTest : public testing::Test
{
 public:
  using IT = std::tuple_element_t<0, Types>;
  using NT = std::tuple_element_t<1, Types>;
  using OT = std::tuple_element_t<2, Types>;
  using Matrix = Distributed<IT, NT, OT>;
};

template <class IT, class NT, class OT, class Matrix>
typename Matrix::MatrixType SampleLocal()
{
  using Local = typename Matrix::MatrixType;
  CooMatrix<IT, NT, OT> coo;
  coo.Allocate(3, 2, 2);
  coo.entries[0] = {0, 0, static_cast<NT>(2)};
  coo.entries[1] = {1, 0, static_cast<NT>(3)};
  coo.entries[2] = {1, 1, static_cast<NT>(4)};
  if constexpr (std::is_same_v<Local, CooMatrix<IT, NT, OT>>) {
    return coo;
  } else if constexpr (std::is_same_v<Local, CsrMatrix<IT, NT, OT>>) {
    Local result = coo.ToCsr();
    return result;
  } else if constexpr (std::is_same_v<Local, CscMatrix<IT, NT, OT>>) {
    Local result = coo.ToCsc();
    return result;
  } else {
    const CscMatrix<IT, NT, OT> csc = coo.ToCsc();
    Local result = Local::FromCsc(csc);
    return result;
  }
}

template <class IT, class NT, class OT, template <class, class, class> class Matrix>
const void* LocalBuffer(const Matrix<IT, NT, OT>& matrix)
{
  if constexpr (std::is_same_v<typename Matrix<IT, NT, OT>::MatrixType, CooMatrix<IT, NT, OT>>) {
    return matrix.local.entries;
  } else {
    return matrix.local.val;
  }
}

template <class IT, class NT, class OT, template <class, class, class> class Matrix>
NT& FirstValue(Matrix<IT, NT, OT>& matrix)
{
  if constexpr (std::is_same_v<typename Matrix<IT, NT, OT>::MatrixType, CooMatrix<IT, NT, OT>>) {
    return matrix.local.entries[0].val;
  } else {
    return matrix.local.val[0];
  }
}

template <class IT, class NT, class OT, class Matrix>
typename Matrix::MatrixType LocalView(const typename Matrix::MatrixType& source)
{
  using Local = typename Matrix::MatrixType;
  if constexpr (std::is_same_v<Local, CooMatrix<IT, NT, OT>>) {
    Local result(source.entries, source.nnz, source.m, source.n);
    return result;
  } else if constexpr (std::is_same_v<Local, CsrMatrix<IT, NT, OT>>) {
    Local result(source.row_ptr, source.col_id, source.val, source.nnz, source.m, source.n);
    return result;
  } else if constexpr (std::is_same_v<Local, CscMatrix<IT, NT, OT>>) {
    Local result(source.col_ptr, source.row_id, source.val, source.nnz, source.m, source.n);
    return result;
  } else {
    OT* pointers = source.col_ptr;
    IT* columns = source.col_id;
    IT* rows = source.row_id;
    NT* values = source.val;
    Local result(pointers, columns, rows, values, source.nnz, source.m, source.n, source.nzc);
    return result;
  }
}

template <class IT, class NT, class OT, template <class, class, class> class Matrix>
void TestAllocate(Matrix<IT, NT, OT>& matrix, OT count, IT m, IT n)
{
  if constexpr (std::is_same_v<typename Matrix<IT, NT, OT>::MatrixType, DcscMatrix<IT, NT, OT>>) {
    const IT stored_columns = count > OT{0} ? IT{1} : IT{0};
    matrix.Allocate(count, m, n, stored_columns);
  } else {
    matrix.Allocate(count, m, n);
  }
}

template <class IT, class NT, class OT, class Matrix>
void CheckDefaultAndEmpty()
{
  // The concrete container owns its local buffers and shares only the grid.
  static_assert(!std::is_copy_constructible_v<Matrix>);
  static_assert(!std::is_copy_assignable_v<Matrix>);
  static_assert(std::is_nothrow_move_constructible_v<Matrix>);
  static_assert(std::is_nothrow_move_assignable_v<Matrix>);
  using LocalMember = typename Matrix::MatrixType Matrix::*;
  using GridMember = typename Matrix::GridPtr Matrix::*;
  using DimensionMember = IT Matrix::*;
  static_assert(std::is_same_v<decltype(&Matrix::local), LocalMember>);
  static_assert(std::is_same_v<decltype(&Matrix::commgrid), GridMember>);
  static_assert(std::is_same_v<decltype(&Matrix::m), DimensionMember>);
  static_assert(std::is_same_v<decltype(&Matrix::n), DimensionMember>);
  Matrix empty;
  EXPECT_EQ(empty.commgrid, nullptr);
  EXPECT_EQ(empty.local.nnz, 0);
  EXPECT_EQ(empty.LocalRows(), 0);
  EXPECT_EQ(empty.LocalCols(), 0);
  Matrix cloned = empty.Clone();
  EXPECT_EQ(cloned.commgrid, nullptr);
  EXPECT_EQ(cloned.local.nnz, 0);
  EXPECT_THROW(empty.Owner(0, 0), std::logic_error);
  EXPECT_THROW(empty.Owns(0, 0), std::logic_error);

  // A shaped empty partition must retain its dimensions through allocation and cloning.
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  Matrix shaped(grid, shape_m, shape_n);
  EXPECT_EQ(shaped.local.m, 2);
  EXPECT_EQ(shaped.local.n, 2);
  EXPECT_EQ(shaped.local.nnz, 0);
  TestAllocate(shaped, OT{0}, shape_m, shape_n);
  Matrix shaped_clone = shaped.Clone();
  EXPECT_EQ(shaped_clone.commgrid, grid);
  EXPECT_EQ(shaped_clone.local.m, 2);
  EXPECT_EQ(shaped_clone.local.n, 2);
  EXPECT_EQ(shaped_clone.local.nnz, 0);

  // Zero dimensions are a valid replacement and participate in global nnz reduction.
  const IT zero_m = 0;
  const IT zero_n = 0;
  TestAllocate(shaped, OT{0}, zero_m, zero_n);
  EXPECT_EQ(shaped.m, 0);
  EXPECT_EQ(shaped.local.m, 0);
  EXPECT_EQ(shaped.local.n, 0);
  EXPECT_EQ(shaped.GlobalNnz(), 0);
}

template <class IT, class NT, class OT, class Matrix>
void CheckCloneAndSharedOwnership()
{
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  typename Matrix::MatrixType data = SampleLocal<IT, NT, OT, Matrix>();
  Matrix matrix(grid, std::move(data), shape_m, shape_n);
  EXPECT_EQ(data.nnz, 0);
  Matrix cloned = matrix.Clone();
  EXPECT_EQ(cloned.commgrid, matrix.commgrid);
  EXPECT_NE(LocalBuffer(cloned), LocalBuffer(matrix));
  EXPECT_EQ(cloned.local.nnz, matrix.local.nnz);
  EXPECT_EQ(FirstValue(cloned), static_cast<NT>(2));
  FirstValue(cloned) = NT{0};
  EXPECT_EQ(FirstValue(matrix), static_cast<NT>(2)) << "Clone must copy local storage";

  // A matrix must keep the communicators alive after the original grid owner is gone.
  std::weak_ptr<const CommGrid2D> lifetime = grid;
  grid.reset();
  EXPECT_FALSE(lifetime.expired());
  EXPECT_EQ(matrix.GlobalNnz(), 3 * matrix.commgrid->Size());
  int row_size = 0;
  MPI_Comm_size(cloned.commgrid->RowWorld(), &row_size);
  EXPECT_EQ(row_size, cloned.commgrid->Cols());
}

template <class IT, class NT, class OT, class Matrix>
void CheckMoves()
{
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  typename Matrix::MatrixType data = SampleLocal<IT, NT, OT, Matrix>();
  Matrix source(grid, std::move(data), shape_m, shape_n);
  const void* buffer = LocalBuffer(source);
  Matrix moved(std::move(source));
  EXPECT_EQ(LocalBuffer(moved), buffer);
  EXPECT_EQ(moved.commgrid, grid);
  EXPECT_EQ(source.commgrid, nullptr);
  EXPECT_EQ(source.local.nnz, 0);
  EXPECT_EQ(source.m, 0);
  Matrix destination(grid, shape_m, shape_n);
  TestAllocate(destination, OT{1}, shape_m, shape_n);
  destination = std::move(moved);
  EXPECT_EQ(LocalBuffer(destination), buffer);
  EXPECT_EQ(moved.commgrid, nullptr);
  EXPECT_EQ(moved.local.nnz, 0);
  Matrix& same = destination;
  destination = std::move(same);
  EXPECT_EQ(LocalBuffer(destination), buffer);
  EXPECT_EQ(destination.commgrid, grid);
}

template <class IT, class NT, class OT, class Matrix>
void CheckViewAndReset()
{
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  typename Matrix::MatrixType owned = SampleLocal<IT, NT, OT, Matrix>();
  {
    typename Matrix::MatrixType view = LocalView<IT, NT, OT, Matrix>(owned);
    Matrix borrowed(grid, std::move(view), shape_m, shape_n);
    EXPECT_FALSE(borrowed.local.memowned);
    Matrix copy = borrowed.Clone();
    EXPECT_TRUE(copy.local.memowned);
    EXPECT_NE(LocalBuffer(copy), LocalBuffer(borrowed));
    borrowed.Reset();
    EXPECT_EQ(borrowed.commgrid, nullptr);
    EXPECT_EQ(borrowed.local.nnz, 0);
    EXPECT_EQ(borrowed.m, 0);
    EXPECT_EQ(FirstValue(copy), static_cast<NT>(2));
  }
  EXPECT_EQ(owned.nnz, 3) << "Destroying a distributed view must not free its backing storage";

  // Preserve an explicit owner before Reset releases a live owning matrix's buffers.
  Matrix matrix(grid, std::move(owned), shape_m, shape_n);
  typename Matrix::MatrixType retained = std::move(matrix.local);
  matrix.Reset();
  EXPECT_EQ(matrix.commgrid, nullptr);
  EXPECT_EQ(retained.nnz, 3);
}

template <class IT, class NT, class OT, class Matrix>
void CheckContiguousIndices()
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int rows = size % 2 == 0 ? 2 : 1;
  const int cols = size / rows;
  const CommGrid2D::RankLayout column_major = [](int rank, int grid_rows, int) {
    return std::pair<int, int>{rank % grid_rows, rank / grid_rows};
  };
  std::shared_ptr<const CommGrid2D> grid =
      std::make_shared<CommGrid2D>(MPI_COMM_WORLD, rows, cols, column_major);
  const IT shape_m = 11;
  const IT shape_n = 13;
  const Matrix matrix(grid, shape_m, shape_n);
  const IT row_extent = shape_m / rows + (shape_m % rows != 0);
  const IT col_extent = shape_n / cols + (shape_n % cols != 0);

  // Enumerate global indices independently of the implementation's closed-form counts.
  std::vector<IT> expected_rows;
  std::vector<IT> expected_cols;
  for (IT row = 0; row < shape_m; ++row) {
    if (row / row_extent == grid->Row()) expected_rows.push_back(row);
  }
  for (IT col = 0; col < shape_n; ++col) {
    if (col / col_extent == grid->Col()) expected_cols.push_back(col);
  }
  EXPECT_EQ(matrix.local.m, expected_rows.size());
  EXPECT_EQ(matrix.local.n, expected_cols.size());
  for (IT row = 0; row < matrix.LocalRows(); ++row) {
    EXPECT_EQ(matrix.GlobalRow(row), expected_rows[row]);
    EXPECT_EQ(matrix.LocalRow(expected_rows[row]), row);
  }
  for (IT col = 0; col < matrix.LocalCols(); ++col) {
    EXPECT_EQ(matrix.GlobalCol(col), expected_cols[col]);
    EXPECT_EQ(matrix.LocalCol(expected_cols[col]), col);
  }
  for (IT row = 0; row < shape_m; ++row) {
    const int process_row = row / row_extent;
    if (process_row != grid->Row()) EXPECT_THROW(matrix.LocalRow(row), std::out_of_range);
    for (IT col = 0; col < shape_n; ++col) {
      const int process_col = col / col_extent;
      const int expected_rank = process_col * rows + process_row;
      EXPECT_EQ(matrix.Owner(row, col), expected_rank) << "Entry (" << row << ", " << col << ")";
      EXPECT_EQ(matrix.Owns(row, col), expected_rank == grid->Rank());
    }
  }
  for (IT col = 0; col < shape_n; ++col) {
    if (col / col_extent != grid->Col()) {
      EXPECT_THROW(matrix.LocalCol(col), std::out_of_range);
    }
  }
  EXPECT_THROW(matrix.GlobalRow(matrix.LocalRows()), std::out_of_range);
  EXPECT_THROW(matrix.GlobalCol(matrix.LocalCols()), std::out_of_range);
  EXPECT_THROW(matrix.LocalRow(shape_m), std::out_of_range);
  EXPECT_THROW(matrix.LocalCol(shape_n), std::out_of_range);
  EXPECT_THROW(matrix.Owner(-1, 0), std::out_of_range);
}

template <class IT, class NT, class OT, class Matrix>
void CheckEmptyRanksAndIntegerLimits()
{
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT tiny_m = 1;
  const IT tiny_n = 1;
  const Matrix small(grid, tiny_m, tiny_n);
  EXPECT_EQ(small.local.m, grid->Row() == 0 ? 1 : 0);
  EXPECT_EQ(small.local.n, grid->Col() == 0 ? 1 : 0);
  EXPECT_EQ(small.GlobalNnz(), 0);

  // COO needs no pointer array for an empty matrix, even at the maximum index.
  // Exercise the shared mapping using only that storage format.
  const IT largest = std::numeric_limits<IT>::max();
  const IT large_m = largest;
  const IT large_n = largest;
  const DistSpCOO<IT, NT, OT> matrix(grid, large_m, large_n);
  if (matrix.LocalRows() != 0) {
    const IT last = matrix.LocalRows() - 1;
    EXPECT_LT(matrix.GlobalRow(last), largest);
    EXPECT_EQ(matrix.LocalRow(matrix.GlobalRow(last)), last);
  }
}

template <class IT, class NT, class OT, class Matrix>
void CheckInvalidInputAndFailedAllocation()
{
  std::shared_ptr<const CommGrid2D> grid = std::make_shared<CommGrid2D>();
  const IT shape_m = 2 * grid->Rows();
  const IT shape_n = 2 * grid->Cols();
  EXPECT_THROW(Matrix(nullptr, shape_m, shape_n), std::invalid_argument);
  const IT negative_m = -1;
  const IT negative_n = 1;
  EXPECT_THROW(Matrix(grid, negative_m, negative_n), std::invalid_argument);
  typename Matrix::MatrixType wrong;
  EXPECT_THROW(Matrix(grid, std::move(wrong), shape_m, shape_n), std::invalid_argument);
  typename Matrix::MatrixType data = SampleLocal<IT, NT, OT, Matrix>();
  Matrix matrix(grid, std::move(data), shape_m, shape_n);
  const void* buffer = LocalBuffer(matrix);
  const IT replacement_m = shape_m + 1;
  const IT replacement_n = shape_n + 1;
  if constexpr (sizeof(OT) == sizeof(int64_t)) {
    const OT unsizeable = std::numeric_limits<OT>::max();
    EXPECT_THROW(TestAllocate(matrix, unsizeable, replacement_m, replacement_n), std::bad_array_new_length);
  } else {
    EXPECT_THROW(TestAllocate(matrix, OT{-1}, replacement_m, replacement_n), std::invalid_argument);
  }
  EXPECT_EQ(LocalBuffer(matrix), buffer);
  EXPECT_EQ(matrix.local.nnz, 3);
  EXPECT_EQ(matrix.m, shape_m);
  EXPECT_EQ(matrix.n, shape_n);
  EXPECT_EQ(matrix.commgrid, grid);
  TestAllocate(matrix, OT{0}, replacement_m, replacement_n);
  EXPECT_EQ(matrix.m, replacement_m);
  EXPECT_EQ(matrix.n, replacement_n);
  EXPECT_EQ(matrix.local.nnz, 0);
}

inline int TestRun(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
