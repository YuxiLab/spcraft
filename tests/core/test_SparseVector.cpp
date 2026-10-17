#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "SpCraft.h"

using namespace spcraft;

template <class Types>
class SparseVectorTest : public ::testing::Test
{
 public:
  using IT = std::tuple_element_t<0, Types>;
  using NT = std::tuple_element_t<1, Types>;
  using Vector = SparseVector<IT, NT>;
  using Dense = DenseVector<IT, NT>;
};
using SparseVectorTypes = ::testing::Types<std::tuple<std::int32_t, float>, std::tuple<std::int32_t, double>,
                                           std::tuple<std::int32_t, bool>, std::tuple<std::int64_t, float>,
                                           std::tuple<std::int64_t, double>, std::tuple<std::int64_t, bool>>;
TYPED_TEST_SUITE(SparseVectorTest, SparseVectorTypes);

//! Length 6 with entries at positions 1, 3, and 4: [ . a . b c . ].
//! Bool values are all true, so the sample is the same pattern in every type.
template <class Vector>
Vector SparseSample()
{
  using IT = decltype(Vector::n);
  using NT = std::remove_pointer_t<decltype(Vector::val)>;
  Vector vector(3, 6);
  const IT positions[] = {1, 3, 4};
  const NT values[] = {NT(2), NT(5), NT(7)};
  std::copy(positions, positions + 3, vector.ind);
  std::copy(values, values + 3, vector.val);
  return vector;
}

TYPED_TEST(SparseVectorTest, DefaultConstructionOwnsNothing)
{
  typename TestFixture::Vector vector;
  EXPECT_EQ(vector.ind, nullptr);
  EXPECT_EQ(vector.val, nullptr);
  EXPECT_EQ(vector.nnz, 0);
  EXPECT_EQ(vector.n, 0);
}

TYPED_TEST(SparseVectorTest, AllocateRecordsShapeAndOwnsStorage)
{
  typename TestFixture::Vector vector;
  vector.Allocate(3, 10);
  EXPECT_NE(vector.ind, nullptr);
  EXPECT_NE(vector.val, nullptr);
  EXPECT_EQ(vector.nnz, 3);
  EXPECT_EQ(vector.n, 10);
  EXPECT_EQ(vector.size(), 10);
  EXPECT_TRUE(vector.memowned);
}

TYPED_TEST(SparseVectorTest, AnEmptyVectorIsRepresentable)
{
  using Vector = typename TestFixture::Vector;
  // nnz == 0 keeps its length: an all-fill vector, not an error.
  Vector vector(0, 8);
  EXPECT_EQ(vector.ind, nullptr);
  EXPECT_EQ(vector.val, nullptr);
  EXPECT_EQ(vector.nnz, 0);
  EXPECT_EQ(vector.n, 8);
  Vector copy = vector.Clone();
  EXPECT_EQ(copy.nnz, 0);
  EXPECT_EQ(copy.n, 8);
  EXPECT_TRUE(copy.memowned);
  Vector zero_length(0, 0);
  EXPECT_EQ(zero_length.n, 0);
}

TYPED_TEST(SparseVectorTest, AllocateReplacesPreviousStorage)
{
  typename TestFixture::Vector vector = SparseSample<typename TestFixture::Vector>();
  vector.Allocate(5, 9);
  EXPECT_EQ(vector.nnz, 5);
  EXPECT_EQ(vector.n, 9);
  EXPECT_TRUE(vector.memowned);
}

TYPED_TEST(SparseVectorTest, ThePointerConstructorIsTheBorrowPath)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using Vector = typename TestFixture::Vector;
  IT positions[] = {0, 2};
  NT values[] = {NT(1), NT(1)};
  {
    Vector view(positions, values, 2, 4);
    EXPECT_FALSE(view.memowned);
    EXPECT_EQ(view.ind, positions);
    EXPECT_EQ(view.val, values);
    // A moved view keeps borrowing, and allocating must not free the caller's arrays.
    Vector moved(std::move(view));
    EXPECT_FALSE(moved.memowned);
    moved.Allocate(1, 4);
    EXPECT_TRUE(moved.memowned);
  }
  EXPECT_EQ(positions[1], 2) << "a destroyed view must leave the caller's arrays alone";
  EXPECT_EQ(values[1], NT(1));

  Vector empty_view(nullptr, nullptr, 0, 5);
  EXPECT_FALSE(empty_view.memowned);
  EXPECT_EQ(empty_view.n, 5);
}

TYPED_TEST(SparseVectorTest, MoveConstructionTransfersAndEmptiesTheSource)
{
  using Vector = typename TestFixture::Vector;
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  Vector source = SparseSample<Vector>();
  const IT* positions = source.ind;
  const NT* values = source.val;
  Vector moved(std::move(source));
  EXPECT_EQ(moved.ind, positions);
  EXPECT_EQ(moved.val, values);
  EXPECT_EQ(moved.nnz, 3);
  EXPECT_EQ(moved.n, 6);
  EXPECT_TRUE(moved.memowned);
  EXPECT_EQ(source.ind, nullptr);
  EXPECT_EQ(source.val, nullptr);
  EXPECT_EQ(source.nnz, 0);
  EXPECT_EQ(source.n, 0);
  EXPECT_FALSE(source.memowned);
}

TYPED_TEST(SparseVectorTest, MoveAssignmentTakesTheSourceStorage)
{
  using Vector = typename TestFixture::Vector;
  using IT = typename TestFixture::IT;
  Vector source = SparseSample<Vector>();
  const IT* positions = source.ind;
  Vector target(4, 20);
  target = std::move(source);
  EXPECT_EQ(target.ind, positions);
  EXPECT_EQ(target.nnz, 3);
  EXPECT_EQ(target.n, 6);
  EXPECT_EQ(source.ind, nullptr);
  EXPECT_FALSE(source.memowned);
}

TYPED_TEST(SparseVectorTest, SelfMoveAssignmentMustNotDestroyTheObject)
{
  using Vector = typename TestFixture::Vector;
  Vector vector = SparseSample<Vector>();
  Vector& alias = vector;
  vector = std::move(alias);
  ASSERT_NE(vector.ind, nullptr);
  EXPECT_EQ(vector.nnz, 3);
  EXPECT_EQ(vector.ind[2], 4);
  EXPECT_TRUE(vector.memowned);
}

TYPED_TEST(SparseVectorTest, CloneIsADeepCopy)
{
  using NT = typename TestFixture::NT;
  using Vector = typename TestFixture::Vector;
  const Vector original = SparseSample<Vector>();
  Vector copy = original.Clone();
  EXPECT_NE(copy.ind, original.ind);
  EXPECT_NE(copy.val, original.val);
  EXPECT_EQ(copy.nnz, original.nnz);
  EXPECT_EQ(copy.n, original.n);
  EXPECT_TRUE(std::equal(original.ind, original.ind + 3, copy.ind));
  EXPECT_TRUE(std::equal(original.val, original.val + 3, copy.val));
  copy.ind[0] = 0;
  copy.val[0] = NT{};
  EXPECT_EQ(original.ind[0], 1) << "writing to a clone must not touch the original";
  EXPECT_EQ(original.val[0], NT(2));
}

TYPED_TEST(SparseVectorTest, ResetReleasesWithoutFreeing)
{
  using IT = typename TestFixture::IT;
  using NT = typename TestFixture::NT;
  using Vector = typename TestFixture::Vector;
  Vector vector = SparseSample<Vector>();
  IT* positions = vector.ind;
  NT* values = vector.val;
  vector.Reset();
  EXPECT_EQ(vector.ind, nullptr);
  EXPECT_EQ(vector.val, nullptr);
  EXPECT_EQ(vector.nnz, 0);
  EXPECT_EQ(vector.n, 0);
  EXPECT_FALSE(vector.memowned);
  std::free(positions);
  std::free(values);
}

TYPED_TEST(SparseVectorTest, ARequestThatCannotBeSizedIsRefused)
{
  using IT = typename TestFixture::IT;
  // count * sizeof(IT) overflows std::size_t for int64 indices; int32 counts
  // are always sizeable, so they check the length bound instead.
  using Vector = typename TestFixture::Vector;
  Vector vector = SparseSample<Vector>();
  const IT* positions = vector.ind;
  constexpr IT kHuge = std::numeric_limits<IT>::max();
  if constexpr (sizeof(IT) == sizeof(std::int64_t)) {
    EXPECT_THROW(vector.Allocate(kHuge, kHuge), std::bad_array_new_length);
  }
  EXPECT_THROW(vector.Allocate(7, 6), std::invalid_argument) << "nnz cannot exceed the length";
  EXPECT_EQ(vector.ind, positions) << "a failed Allocate must leave the vector untouched";
  EXPECT_EQ(vector.nnz, 3);
  EXPECT_EQ(vector.n, 6);
}

TYPED_TEST(SparseVectorTest, NegativeSizesAreRejected)
{
  using IT = typename TestFixture::IT;
  typename TestFixture::Vector vector;
  EXPECT_THROW(vector.Allocate(IT{-1}, IT{4}), std::invalid_argument);
  EXPECT_THROW(vector.Allocate(IT{0}, IT{-1}), std::invalid_argument);
  EXPECT_EQ(vector.ind, nullptr);
}

TYPED_TEST(SparseVectorTest, ToDenseScattersEntriesOverTheFill)
{
  using NT = typename TestFixture::NT;
  using Vector = typename TestFixture::Vector;
  using Dense = typename TestFixture::Dense;
  const Vector sparse = SparseSample<Vector>();
  const Dense dense = sparse.ToDense(NT{});
  ASSERT_EQ(dense.n, 6);
  const NT expected[] = {NT{}, NT(2), NT{}, NT(5), NT(7), NT{}};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(dense.val[i], expected[i]) << "position " << i;
  }
}

TYPED_TEST(SparseVectorTest, FromDenseRoundTrips)
{
  using NT = typename TestFixture::NT;
  using Vector = typename TestFixture::Vector;
  using Dense = typename TestFixture::Dense;
  const Vector original = SparseSample<Vector>();
  const Dense dense = original.ToDense(NT{});
  const Vector sparse = Vector::FromDense(dense, NT{});
  ASSERT_EQ(sparse.nnz, 3);
  EXPECT_EQ(sparse.n, 6);
  EXPECT_TRUE(std::equal(original.ind, original.ind + 3, sparse.ind));
  EXPECT_TRUE(std::equal(original.val, original.val + 3, sparse.val));
}

TEST(SparseVector, ANonzeroFillIsTheBackgroundValue)
{
  // [ 9 9 4 9 ] with fill 9 stores one entry; expanding with fill 9 restores it.
  DenseVector<std::int32_t, double> dense(4, 9.0);
  dense.val[2] = 4.0;
  const SparseVector<std::int32_t, double> sparse = SparseVector<std::int32_t, double>::FromDense(dense, 9.0);
  ASSERT_EQ(sparse.nnz, 1);
  EXPECT_EQ(sparse.ind[0], 2);
  EXPECT_EQ(sparse.val[0], 4.0);
  const DenseVector<std::int32_t, double> restored = sparse.ToDense(9.0);
  EXPECT_TRUE(std::equal(dense.val, dense.val + 4, restored.val));
}

TEST(SparseVector, AnAllFillVectorHasNoEntries)
{
  DenseVector<std::int64_t, double> dense(5, 0.0);
  const SparseVector<std::int64_t, double> sparse = SparseVector<std::int64_t, double>::FromDense(dense, 0.0);
  EXPECT_EQ(sparse.nnz, 0);
  EXPECT_EQ(sparse.n, 5);
  const DenseVector<std::int64_t, double> restored = sparse.ToDense(0.0);
  ASSERT_EQ(restored.n, 5);
  EXPECT_TRUE(std::all_of(restored.val, restored.val + 5, [](double v) { return v == 0.0; }));
}

TEST(SparseVector, ToDenseRejectsIndicesBreakingTheInvariant)
{
  using Vector = SparseVector<std::int32_t, double>;
  double values[] = {1.0, 2.0};
  std::int32_t unsorted[] = {3, 1};
  std::int32_t duplicated[] = {2, 2};
  std::int32_t past_end[] = {1, 4};
  std::int32_t negative[] = {-1, 2};
  constexpr std::int32_t kLength = 4;
  for (std::int32_t* positions : {unsorted, duplicated, past_end, negative}) {
    // Each pair breaks one condition; the message names the pair that did not throw.
    const Vector view(positions, values, 2, kLength);
    EXPECT_THROW((void)view.ToDense(0.0), std::invalid_argument) << positions[0] << ", " << positions[1];
  }
}

class SparseVectorThreadTest : public ::testing::TestWithParam<int>
{
 protected:
  void SetUp() override { OMP_SET_NUM_THREADS(GetParam()); }
};

TEST_P(SparseVectorThreadTest, ParallelConversionsMatchEigen)
{
  using Vector = SparseVector<std::int64_t, double>;
  using Dense = DenseVector<std::int64_t, double>;
  // Long enough that every thread of the team receives a share of each loop.
  constexpr std::int64_t kLength = 10000;
  constexpr int kSeed = 2024;
  constexpr std::int64_t kZeroStride = 3;  // Zero every third element so the pattern is sparse.

  // Build the dense input.
  Dense dense(kLength);
  dense.Random(kSeed);
  for (std::int64_t i = 0; i < kLength; i += kZeroStride) dense.val[i] = 0.0;

  // Eigen selects the same nonzeros independently of SpCraft's compaction.
  const Vector sparse = Vector::FromDense(dense, 0.0);
  const Eigen::Map<const Eigen::VectorXd> reference_dense(dense.val, kLength);
  const Eigen::SparseVector<double> reference = reference_dense.sparseView();
  ASSERT_EQ(sparse.nnz, reference.nonZeros());
  std::int64_t k = 0;
  for (Eigen::SparseVector<double>::InnerIterator it(reference); it; ++it, ++k) {
    EXPECT_EQ(sparse.ind[k], it.index()) << "entry " << k;
    EXPECT_EQ(sparse.val[k], it.value()) << "entry " << k;
  }

  // The parallel scatter and copy reproduce the input exactly.
  const Dense restored = sparse.ToDense(0.0);
  EXPECT_TRUE(std::equal(dense.val, dense.val + kLength, restored.val));
  const Vector copy = sparse.Clone();
  EXPECT_TRUE(std::equal(sparse.ind, sparse.ind + sparse.nnz, copy.ind));
  EXPECT_TRUE(std::equal(sparse.val, sparse.val + sparse.nnz, copy.val));

  // One out-of-order pair near the end must reach the reduction from whichever thread owns it.
  Vector broken = sparse.Clone();
  std::swap(broken.ind[broken.nnz - 2], broken.ind[broken.nnz - 1]);
  EXPECT_THROW((void)broken.ToDense(0.0), std::invalid_argument);
}

const std::vector<int> kThreadCounts =
    SPCRAFT_OPENMP_ENABLED ? std::vector<int>{1, 2, 4} : std::vector<int>{1};
INSTANTIATE_TEST_SUITE_P(OpenMPTeams, SparseVectorThreadTest, ::testing::ValuesIn(kThreadCounts));
