#include "SpCraft.h"

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpi.h>

using namespace spcraft;

constexpr int kLocalExtent = 2;

template <class NT>
using ReferenceMatrix = Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic>;

template <class IT, class NT, class OT>
DcscMatrix<IT, NT, OT> ExtractBlock(const ReferenceMatrix<NT>& matrix, const int* coordinates)
{
  const int first_row = coordinates[0] * kLocalExtent;
  const int first_column = coordinates[1] * kLocalExtent;
  OT nonzeros = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      if (matrix(first_row + row, first_column + column) != NT{0}) ++nonzeros;
    }
  }

  CooMatrix<IT, NT, OT> coo;
  coo.Allocate(nonzeros, kLocalExtent, kLocalExtent);
  OT position = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      const NT value = matrix(first_row + row, first_column + column);
      if (value != NT{0}) {
        coo.entries[position++] = {static_cast<IT>(row), static_cast<IT>(column), value};
      }
    }
  }
  const CscMatrix<IT, NT, OT> csc = coo.ToCsc();
  DcscMatrix<IT, NT, OT> block = DcscMatrix<IT, NT, OT>::FromCsc(csc);
  return block;
}

template <class IT, class NT, class OT>
void CheckProduct(const ReferenceMatrix<NT>& a, const ReferenceMatrix<NT>& b, MPI_Comm grid)
{
  int rank = 0;
  int coordinates[2] = {};
  MPI_Comm_rank(grid, &rank);
  MPI_Cart_coords(grid, rank, 2, coordinates);
  SCOPED_TRACE(testing::Message() << "Cartesian rank " << rank << ", block (" << coordinates[0]
                                  << ", " << coordinates[1] << ")");

  const DcscMatrix<IT, NT, OT> local_a = ExtractBlock<IT, NT, OT>(a, coordinates);
  const DcscMatrix<IT, NT, OT> local_b = ExtractBlock<IT, NT, OT>(b, coordinates);
  const CooMatrix<IT, NT, OT> actual = Mpi2DSummaSpGEMM<PlusTimesRing<NT>>(local_a, local_b, grid);

  // Every rank computes an independent global Eigen reference, then compares
  // its own block. A local product alone misses contributions from other ranks.
  const ReferenceMatrix<NT> expected = a * b;
  const int first_row = coordinates[0] * kLocalExtent;
  const int first_column = coordinates[1] * kLocalExtent;
  OT expected_nonzeros = 0;
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      if (expected(first_row + row, first_column + column) != NT{0}) ++expected_nonzeros;
    }
  }

  EXPECT_EQ(actual.m, kLocalExtent);
  EXPECT_EQ(actual.n, kLocalExtent);
  EXPECT_EQ(actual.nnz, expected_nonzeros) << "Output block has the wrong nonzero count";
  ASSERT_GE(actual.nnz, OT{0});
  if (actual.nnz != OT{0}) ASSERT_NE(actual.entries, nullptr);

  ReferenceMatrix<NT> values = ReferenceMatrix<NT>::Zero(kLocalExtent, kLocalExtent);
  for (OT position = 0; position < actual.nnz; ++position) {
    const TupleEntry<IT, NT>& entry = actual.entries[position];
    ASSERT_GE(entry.row, IT{0});
    ASSERT_LT(entry.row, kLocalExtent);
    ASSERT_GE(entry.col, IT{0});
    ASSERT_LT(entry.col, kLocalExtent);
    values(entry.row, entry.col) += entry.val;
  }

  constexpr NT kTolerance = std::is_same_v<NT, float> ? NT{1e-6} : NT{1e-12};
  for (int row = 0; row < kLocalExtent; ++row) {
    for (int column = 0; column < kLocalExtent; ++column) {
      EXPECT_NEAR(values(row, column), expected(first_row + row, first_column + column), kTolerance)
          << "Wrong value at local (" << row << ", " << column << ")";
    }
  }
}

template <class Types>
class Summa2DTest : public testing::Test
{
 protected:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  MPI_Comm grid = MPI_COMM_NULL;
  int width = 0;

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  void SetUp() override
  {
    int processes = 0;
    int dimensions[2] = {};
    const int periods[2] = {};
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Dims_create(processes, 2, dimensions);
    ASSERT_EQ(dimensions[0], dimensions[1]) << "Run with a square number of MPI ranks";
    width = dimensions[0];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &grid);
  }

  void TearDown() override
  {
    if (grid != MPI_COMM_NULL) MPI_Comm_free(&grid);
  }
};

using SummaTypes =
    testing::Types<std::tuple<int32_t, float, int32_t>, std::tuple<int32_t, double, int32_t>,
                   std::tuple<int32_t, float, int64_t>, std::tuple<int32_t, double, int64_t>,
                   std::tuple<int64_t, float, int64_t>, std::tuple<int64_t, double, int64_t>>;
TYPED_TEST_SUITE(Summa2DTest, SummaTypes);

TYPED_TEST(Summa2DTest, NonemptyProductMatchesEigen)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  const int extent = kLocalExtent * this->width;
  ReferenceMatrix<NT> a = ReferenceMatrix<NT>::Zero(extent, extent);
  ReferenceMatrix<NT> b = ReferenceMatrix<NT>::Zero(extent, extent);
  // A = 2I + cyclic forward shift; B = 3I + cyclic backward shift.
  // On a 2x2 process grid their product includes contributions across blocks.
  for (int row = 0; row < extent; ++row) {
    a(row, row) = NT{2};
    a(row, (row + 1) % extent) = NT{1};
    b(row, row) = NT{3};
    b(row, (row + extent - 1) % extent) = NT{1};
  }
  CheckProduct<IT, NT, OT>(a, b, this->grid);
}

TYPED_TEST(Summa2DTest, RemoteOperandsReachAnInitiallyEmptyBlock)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  const int extent = kLocalExtent * this->width;
  ReferenceMatrix<NT> a = ReferenceMatrix<NT>::Zero(extent, extent);
  ReferenceMatrix<NT> b = ReferenceMatrix<NT>::Zero(extent, extent);
  a(0, extent - 1) = NT{2};
  b(extent - 1, 0) = NT{3};
  // With four ranks, block (0, 0) starts with empty local operands but must
  // produce C(0, 0) = 6 after receiving the remote panels.
  CheckProduct<IT, NT, OT>(a, b, this->grid);
}

TYPED_TEST(Summa2DTest, EmptyProductKeepsLocalShape)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  const int extent = kLocalExtent * this->width;
  const ReferenceMatrix<NT> empty = ReferenceMatrix<NT>::Zero(extent, extent);
  CheckProduct<IT, NT, OT>(empty, empty, this->grid);
}

TYPED_TEST(Summa2DTest, RejectsIncompatibleInnerDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  DcscMatrix<IT, NT, OT> a;
  DcscMatrix<IT, NT, OT> b;
  a.Allocate(0, kLocalExtent, kLocalExtent + 1, 0);
  b.Allocate(0, kLocalExtent, kLocalExtent, 0);
  using Ring = PlusTimesRing<NT>;
  EXPECT_THROW(static_cast<void>(Mpi2DSummaSpGEMM<Ring>(a, b, this->grid)), std::invalid_argument);
}

int TestRun(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    MPI_Finalize();
    return 1;
  }
  const int status = TestRun(argc, argv);
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
}
