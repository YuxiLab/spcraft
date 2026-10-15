#include "SpCraft.h"

#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

using namespace spcraft;

template <class NT>
class MpiUtilTest : public testing::Test
{
};

using MpiScalarTypes = testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t,
                                      uint64_t, float, double, bool>;
TYPED_TEST_SUITE(MpiUtilTest, MpiScalarTypes);

TYPED_TEST(MpiUtilTest, ScalarExtentAndQualifiedTypesMatch)
{
  using NT = TypeParam;
  const MPI_Datatype datatype = MPIType<NT>();
  int bytes = 0;
  MPI_Aint lower = 0;
  MPI_Aint extent = 0;
  EXPECT_EQ(MPI_Type_size(datatype, &bytes), MPI_SUCCESS);
  EXPECT_EQ(bytes, sizeof(NT));
  EXPECT_EQ(MPI_Type_get_extent(datatype, &lower, &extent), MPI_SUCCESS);
  EXPECT_EQ(lower, 0);
  EXPECT_EQ(extent, sizeof(NT));
  EXPECT_EQ(MPIType<const NT>(), datatype);
  EXPECT_EQ(MPIType<volatile NT>(), datatype);
}

TYPED_TEST(MpiUtilTest, ScalarTransferPreservesValues)
{
  using NT = TypeParam;
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int dst = (rank + 1) % size;
  const int src = (rank + size - 1) % size;
  const NT sent = std::is_same_v<NT, bool> ? static_cast<NT>(rank % 2) : static_cast<NT>(rank + 1);
  const NT expected = std::is_same_v<NT, bool> ? static_cast<NT>(src % 2) : static_cast<NT>(src + 1);
  NT recv{};
  const MPI_Datatype type = MPIType<NT>();
  MPI_Status status{};
  constexpr int tag = 0;
  const MPI_Comm comm = MPI_COMM_WORLD;
  const int error = MPI_Sendrecv(&sent, 1, type, dst, tag, &recv, 1, type, src, tag, comm, &status);
  EXPECT_EQ(error, MPI_SUCCESS);
  EXPECT_EQ(recv, expected) << "Scalar transferred from rank " << src;
}

TYPED_TEST(MpiUtilTest, NativeReductionHasScalarSemantics)
{
  using NT = TypeParam;
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  NT total{};
  if constexpr (std::is_same_v<NT, bool>) {
    const NT value = rank == 0;
    const int error = MPI_Allreduce(&value, &total, 1, MPIType<NT>(), MPI_LOR, MPI_COMM_WORLD);
    EXPECT_EQ(error, MPI_SUCCESS);
    EXPECT_TRUE(total);
  } else {
    const NT value = static_cast<NT>(rank + 1);
    const NT expected = static_cast<NT>(static_cast<int64_t>(size) * (size + 1) / 2);
    const int error = MPI_Allreduce(&value, &total, 1, MPIType<NT>(), MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(error, MPI_SUCCESS);
    EXPECT_EQ(total, expected);
  }
}

int TestRun(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
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
