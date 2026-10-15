#include "DistMatrixTest.h"

using namespace spcraft;

template <class Types>
using DistSpCSCTest = DistributedMatrixTest<DistSpCSC, Types>;

TYPED_TEST_SUITE(DistSpCSCTest, DistributedTypes);

TYPED_TEST(DistSpCSCTest, DefaultAndEmpty)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;
  CheckDefaultAndEmpty<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, CloneAndSharedGridOwnership)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckCloneAndSharedOwnership<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, MoveConstructionAssignmentAndSelfMove)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckMoves<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, ViewsAndReset)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;
  CheckViewAndReset<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, ContiguousIndicesMatchGlobalEnumeration)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckContiguousIndices<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, EmptyRanksAndLargeIndices)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckEmptyRanksAndIntegerLimits<IT, NT, OT, Matrix>();
}

TYPED_TEST(DistSpCSCTest, InvalidInputAndStrongAllocationGuarantee)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using OT = typename TestFixture::OT;
  using Matrix = typename TestFixture::Matrix;

  CheckInvalidInputAndFailedAllocation<IT, NT, OT, Matrix>();
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  const int status = TestRun(argc, argv);
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
}
