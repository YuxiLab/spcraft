#include "SpCraft.h"

#include <memory>

#include "SummaTestCommon.h"

using namespace spcraft;

template <class Matrix>
class Summa3DTest : public testing::Test
{
 protected:
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::unique_ptr<CommGrid3D> grid{};  //!< Owns the layered grid for this case.

  /*****************************************************************************
   *                             Member Functions                              *
   ****************************************************************************/
  void SetUp() override
  {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int layers = size == 8 ? 2 : size;
    grid = std::make_unique<CommGrid3D>(MPI_COMM_WORLD, layers);
  }
};
TYPED_TEST_SUITE(Summa3DTest, DistributedSummaTypes);

template <class IT, class NT, class OT, class Matrix>
void CheckLayeredProduct(const CommGrid3D& grid, const Ref<NT>& a, const Ref<NT>& b)
{
  const IT a_shape_m = static_cast<IT>(a.rows());
  const IT a_shape_n = static_cast<IT>(a.cols());
  const IT b_shape_m = static_cast<IT>(b.rows());
  const IT b_shape_n = static_cast<IT>(b.cols());
  int rank = 0;
  MPI_Comm_rank(grid.World(), &rank);
  const int side = grid.Plane()->Rows();
  const IT row_block = a_shape_m / side + (a_shape_m % side != 0);
  const IT inner_block = a_shape_n / side + (a_shape_n % side != 0);
  const IT col_block = b_shape_n / side + (b_shape_n % side != 0);
  const auto owns_a = [&](IT row, IT col) { return grid.AOwner(row / row_block, col / inner_block) == rank; };
  const auto owns_b = [&](IT row, IT col) { return grid.BOwner(row / inner_block, col / col_block) == rank; };
  const Matrix da = MakeSummaInput<IT, NT, OT, Matrix>(a, grid.Plane(), a_shape_m, a_shape_n, owns_a);
  const Matrix db = MakeSummaInput<IT, NT, OT, Matrix>(b, grid.Plane(), b_shape_m, b_shape_n, owns_b);

  // Every layer receives only its own inner blocks; no full input is replicated.
  int64_t local_counts[2] = {da.local.nnz, db.local.nnz};
  int64_t global_counts[2] = {};
  MPI_Allreduce(local_counts, global_counts, 2, MPI_INT64_T, MPI_SUM, grid.World());
  EXPECT_EQ(global_counts[0], (a.array() != NT{0}).count());
  EXPECT_EQ(global_counts[1], (b.array() != NT{0}).count());
  const DistSpCOO<IT, NT, OT> result = Mpi3DSummaSpGEMM<PlusTimesRing<NT>>(da, db, grid);
  if (grid.Layer() == 0) {
    EXPECT_EQ(result.commgrid, grid.Plane());
    const Ref<NT> expected = a * b;
    CheckSummaValues(result, expected);
  } else {
    EXPECT_EQ(result.commgrid, nullptr);
    EXPECT_EQ(result.local.entries, nullptr);
    EXPECT_EQ(result.local.nnz, 0);
    EXPECT_EQ(result.m, 0);
    EXPECT_EQ(result.n, 0);
  }
}

TYPED_TEST(Summa3DTest, MultipleLayersAndContiguousEdgesMatchEigen)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  Ref<NT> a = Ref<NT>::Zero(5, 11);
  Ref<NT> b = Ref<NT>::Zero(11, 7);
  for (int i = 0; i < a.rows(); ++i)
    for (int k = 0; k < a.cols(); ++k)
      if ((i + k) % 3 != 0) a(i, k) = static_cast<NT>((i + k) % 5 - 2);
  for (int k = 0; k < b.rows(); ++k)
    for (int j = 0; j < b.cols(); ++j)
      if ((k + j) % 2 == 0) b(k, j) = static_cast<NT>((k + j) % 4 + 1);
  CheckLayeredProduct<IT, NT, OT, Matrix>(*this->grid, a, b);
}

TYPED_TEST(Summa3DTest, EmptyLayersAndLocalPartitions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const Ref<NT> a = Ref<NT>::Constant(1, 1, NT{2});
  const Ref<NT> b = Ref<NT>::Constant(1, 1, NT{3});
  CheckLayeredProduct<IT, NT, OT, Matrix>(*this->grid, a, b);
}

TYPED_TEST(Summa3DTest, ZeroInnerDimension)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const Ref<NT> a(5, 0);
  const Ref<NT> b(0, 7);
  CheckLayeredProduct<IT, NT, OT, Matrix>(*this->grid, a, b);
}

TYPED_TEST(Summa3DTest, ZeroOuterDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const Ref<NT> a = Ref<NT>::Zero(0, 5);
  const Ref<NT> b = Ref<NT>::Zero(5, 3);
  CheckLayeredProduct<IT, NT, OT, Matrix>(*this->grid, a, b);
  const Ref<NT> c = Ref<NT>::Zero(3, 5);
  const Ref<NT> d = Ref<NT>::Zero(5, 0);
  CheckLayeredProduct<IT, NT, OT, Matrix>(*this->grid, c, d);
}

TYPED_TEST(Summa3DTest, ResultKeepsPlaneAliveAfterGridDestruction)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const bool root_layer = this->grid->Layer() == 0;
  std::weak_ptr<const CommGrid2D> lifetime = this->grid->Plane();
  DistSpCOO<IT, NT, OT> result;
  {
    const IT shape_m = 3;
    const IT shape_n = 3;
    const Matrix a(this->grid->Plane(), shape_m, shape_n);
    result = Mpi3DSummaSpGEMM<PlusTimesRing<NT>>(a, a, *this->grid);
  }
  this->grid.reset();
  EXPECT_EQ(lifetime.expired(), !root_layer);
  if (root_layer)
    EXPECT_EQ(result.GlobalNnz(), 0);
  else
    EXPECT_EQ(result.commgrid, nullptr);
}

TYPED_TEST(Summa3DTest, CancellationAcrossLayersRetainsStructure)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  const IT a_shape_m = 1;
  const IT a_shape_n = 2;
  const IT b_shape_m = 2;
  const IT b_shape_n = 1;
  Ref<NT> a(1, 2);
  a << NT{2}, NT{-2};
  const Ref<NT> b = Ref<NT>::Ones(2, 1);
  int rank = 0;
  MPI_Comm_rank(this->grid->World(), &rank);
  const int side = this->grid->Plane()->Rows();
  const IT inner_block = 2 / side + (2 % side != 0);
  const auto owns_a = [&](IT row, IT col) { return this->grid->AOwner(row, col / inner_block) == rank; };
  const auto owns_b = [&](IT row, IT col) { return this->grid->BOwner(row / inner_block, col) == rank; };
  const Matrix da = MakeSummaInput<IT, NT, OT, Matrix>(a, this->grid->Plane(), a_shape_m, a_shape_n, owns_a);
  const Matrix db = MakeSummaInput<IT, NT, OT, Matrix>(b, this->grid->Plane(), b_shape_m, b_shape_n, owns_b);
  const CommGrid3D& grid = *this->grid;
  const DistSpCOO<IT, NT, OT> result = Mpi3DSummaSpGEMM<PlusTimesRing<NT>>(da, db, grid);
  if (grid.Layer() == 0) {
    EXPECT_EQ(result.GlobalNnz(), 1);
    const Ref<NT> expected = a * b;
    CheckSummaValues(result, expected);
  } else {
    EXPECT_EQ(result.commgrid, nullptr);
    EXPECT_EQ(result.local.entries, nullptr);
    EXPECT_EQ(result.local.nnz, 0);
    EXPECT_EQ(result.m, 0);
    EXPECT_EQ(result.n, 0);
  }
}

TYPED_TEST(Summa3DTest, RejectsShapeErrorOnOnlyOneRankCollectively)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Matrix = std::tuple_element_t<3, TypeParam>;

  int rank = 0;
  MPI_Comm_rank(this->grid->World(), &rank);
  const IT inner = rank == 0 ? 3 : 4;
  const IT a_shape_m = 4;
  const IT a_shape_n = inner;
  const IT b_shape_m = 4;
  const IT b_shape_n = 4;
  const Matrix a(this->grid->Plane(), a_shape_m, a_shape_n);
  const Matrix b(this->grid->Plane(), b_shape_m, b_shape_n);
  using Ring = PlusTimesRing<NT>;
  EXPECT_THROW(static_cast<void>(Mpi3DSummaSpGEMM<Ring>(a, b, *this->grid)), std::invalid_argument);
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
  int status = provided < MPI_THREAD_FUNNELED ? 1 : TestRun(argc, argv);
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
}
