#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

using Vector = spcraft::DenseVector<std::int32_t, double>;

Vector Sample()
{
  Vector vector(4);
  for (std::int32_t index = 0; index < 4; ++index) {
    vector.val[index] = static_cast<double>(index) + 1.0;
  }
  return vector;
}

}  // namespace

TEST(DenseVector, DefaultConstructionOwnsNothing)
{
  // --- default construction owns nothing ---------------------------------

  Vector vector;
  EXPECT_TRUE((vector.val == nullptr && vector.n == 0))
      << "a default-constructed vector should hold no buffer";
}

TEST(DenseVector, TheSizingConstructorAllocates)
{
  // --- the sizing constructor allocates ----------------------------------

  Vector vector(6);
  EXPECT_TRUE((vector.n == 6 && vector.val != nullptr)) << "the sizing constructor should allocate";
  EXPECT_TRUE((vector.memowned)) << "the sizing constructor should take ownership";
}

TEST(DenseVector, AnEmptyVectorIsRepresentable)
{
  // --- an empty vector is representable ----------------------------------

  Vector vector(0);
  EXPECT_TRUE((vector.n == 0)) << "a zero-length vector should be allowed";
  Vector copy = vector.Clone();
  EXPECT_TRUE((copy.n == 0)) << "cloning an empty vector should work";
}

TEST(DenseVector, AllocateReplacesPreviousStorage)
{
  // --- Allocate replaces previous storage --------------------------------

  Vector vector(4);
  vector.Allocate(9);
  EXPECT_TRUE((vector.n == 9 && vector.memowned)) << "a second Allocate should replace the first";
}

TEST(DenseVector, ThePointerConstructorIsTheBorrowPath)
{
  // --- the pointer constructor is the borrow path ------------------------

  double buffer[3] = {1.0, 2.0, 3.0};
  {
    Vector view(buffer, 3);
    EXPECT_TRUE((!view.memowned)) << "the pointer constructor must not take ownership";
    EXPECT_TRUE((view.val == buffer)) << "a view should alias the caller's buffer";
  }
  EXPECT_TRUE((buffer[0] == 1.0)) << "a destroyed view must leave the caller's buffer alone";
}

TEST(DenseVector, EmptyBorrowedView)
{
  // A pointer constructor borrows storage, including a legitimate empty view.
  Vector view(nullptr, 0);
  EXPECT_EQ(view.val, nullptr);
  EXPECT_EQ(view.n, 0);
  EXPECT_FALSE(view.memowned);
}

TEST(DenseVector, MoveConstructionAndAssignment)
{
  // --- move construction and assignment ----------------------------------

  Vector source = Sample();
  const auto* buffer = source.val;
  Vector moved(std::move(source));
  EXPECT_TRUE((moved.val == buffer && moved.n == 4)) << "move construction should steal the buffer";
  EXPECT_TRUE((source.val == nullptr && source.n == 0 && !source.memowned))
      << "a moved-from vector must release ownership";

  Vector target(10);
  target = std::move(moved);
  EXPECT_TRUE((target.val == buffer && target.n == 4))
      << "move assignment should take the new storage";
  EXPECT_TRUE((moved.val == nullptr && !moved.memowned))
      << "move assignment should empty the source";
}

TEST(DenseVector, SelfMoveAssignmentMustNotDestroyTheObject)
{
  // --- self-move assignment must not destroy the object ------------------

  Vector vector = Sample();
  Vector& alias = vector;
  vector = std::move(alias);
  EXPECT_TRUE((vector.n == 4 && vector.val != nullptr)) << "self-move must leave the vector intact";
  EXPECT_TRUE((vector.val[0] == 1.0)) << "self-move must not corrupt the values";
}

TEST(DenseVector, CloneIsADeepCopy)
{
  // --- Clone is a deep copy ----------------------------------------------

  Vector original = Sample();
  Vector copy = original.Clone();
  EXPECT_TRUE((copy.val != original.val)) << "Clone must not alias the source";
  EXPECT_TRUE((copy.n == original.n)) << "Clone should preserve the length";
  EXPECT_TRUE((std::equal(original.val, original.val + 4, copy.val)))
      << "Clone should copy the values";
  copy.val[0] = 99.0;
  EXPECT_TRUE((original.val[0] == 1.0)) << "writing to a clone must not touch the original";
}

TEST(DenseVector, ResetReleasesWithoutFreeing)
{
  // --- Reset releases without freeing ------------------------------------

  Vector vector = Sample();
  auto* buffer = vector.val;
  vector.Reset();
  EXPECT_TRUE((vector.val == nullptr && vector.n == 0 && !vector.memowned))
      << "Reset should null every member and drop ownership";
  std::free(buffer);
}

TEST(DenseVector, RandomIsReproducibleAndStaysInRange)
{
  // --- Random is reproducible and stays in range -------------------------

  Vector first(64);
  Vector second(64);
  first.Random(1234);
  second.Random(1234);
  EXPECT_TRUE((std::equal(first.val, first.val + 64, second.val)))
      << "the same seed must produce the same values";
  EXPECT_TRUE((std::all_of(first.val, first.val + 64, [](double v) {
    return v >= -1.0 && v < 1.0;
  }))) << "real values must land in [-1, 1)";

  Vector third(64);
  third.Random(4321);
  EXPECT_TRUE((!std::equal(first.val, first.val + 64, third.val)))
      << "a different seed should produce different values";
}

TEST(DenseVector, ARequestThatCannotBeSizedIsRefused)
{
  // --- a request that cannot be sized is refused, not wrapped ------------

  // size * sizeof(double) overflows std::size_t, so the multiplication would
  // wrap and hand back a buffer far smaller than requested.
  spcraft::DenseVector<std::int64_t, double> vector;
  bool threw = false;
  try {
    vector.Allocate(std::numeric_limits<std::int64_t>::max());
  } catch (const std::bad_array_new_length&) {
    threw = true;
  }
  EXPECT_TRUE((threw)) << "an unsizeable allocation must throw std::bad_array_new_length";
  EXPECT_TRUE((vector.n == 0 && vector.val == nullptr))
      << "a failed Allocate must leave the vector untouched";
}

TEST(DenseVector, NegativeSizesAreRejected)
{
  // --- negative sizes are rejected ---------------------------------------

  Vector vector;
  bool threw = false;
  try {
    vector.Allocate(-1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_TRUE((threw)) << "a negative size must be rejected";
}

template <class Types>
class DenseVectorTypeTest : public ::testing::Test
{
};

using DenseVectorTypes =
    ::testing::Types<std::tuple<std::int32_t, float>, std::tuple<std::int32_t, double>,
                     std::tuple<std::int32_t, bool>, std::tuple<std::int64_t, float>,
                     std::tuple<std::int64_t, double>, std::tuple<std::int64_t, bool>>;
TYPED_TEST_SUITE(DenseVectorTypeTest, DenseVectorTypes);

TYPED_TEST(DenseVectorTypeTest, CompiledConstructorsAndOwnership)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using TypedVector = spcraft::DenseVector<IT, NT>;
  constexpr IT kSize = 4;

  TypedVector original(kSize, NT{1});
  ASSERT_NE(original.val, nullptr);
  EXPECT_EQ(original.size(), kSize);
  EXPECT_TRUE(std::all_of(original.val, original.val + kSize, [](NT v) { return v == NT{1}; }));
  const auto& const_original = original;
  EXPECT_EQ(const_original[0], NT{1});

  auto clone = original.Clone();
  ASSERT_NE(clone.val, nullptr);
  EXPECT_NE(clone.val, original.val);
  clone[0] = NT{};
  EXPECT_EQ(original[0], NT{1}) << "changing the clone must leave its source intact";
  auto* storage = clone.val;
  TypedVector moved(std::move(clone));
  EXPECT_EQ(moved.val, storage);
  EXPECT_EQ(clone.val, nullptr);
  EXPECT_FALSE(clone.memowned);
  TypedVector target(kSize);
  target = std::move(moved);
  EXPECT_EQ(target.val, storage);
  EXPECT_EQ(moved.val, nullptr);
  EXPECT_FALSE(moved.memowned);
  auto& self = target;
  target = std::move(self);
  EXPECT_EQ(target.val, storage);
  EXPECT_EQ(target[0], NT{});

  EXPECT_THROW(target.Allocate(IT{-1}), std::invalid_argument);
  EXPECT_EQ(target.val, storage) << "failed allocation must preserve the owned buffer";
  target.Reset();
  EXPECT_EQ(target.val, nullptr);
  EXPECT_EQ(target.n, IT{0});
  EXPECT_FALSE(target.memowned);
  std::free(storage);

  // A moved view must keep borrowing, and allocating must not free its stack buffer.
  NT borrowed[] = {NT{}, NT{1}};
  TypedVector view(borrowed, 2);
  TypedVector moved_view(std::move(view));
  EXPECT_EQ(moved_view.val, borrowed);
  EXPECT_FALSE(moved_view.memowned);
  moved_view.Allocate(kSize);
  EXPECT_TRUE(moved_view.memowned);
  EXPECT_EQ(borrowed[1], NT{1});

  TypedVector empty(0);
  auto empty_clone = empty.Clone();
  EXPECT_EQ(empty_clone.val, nullptr);
  EXPECT_EQ(empty_clone.n, IT{0});
  EXPECT_TRUE(empty_clone.memowned);
}

TYPED_TEST(DenseVectorTypeTest, CompiledRandomIsDeterministicAndInRange)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using TypedVector = spcraft::DenseVector<IT, NT>;
  constexpr IT kSize = 64;
  constexpr int kSeed = 1234;
  TypedVector first(kSize), second(kSize);
  first.Random(kSeed);
  second.Random(kSeed);
  EXPECT_TRUE(std::equal(first.val, first.val + kSize, second.val));
  if constexpr (std::is_same_v<NT, bool>) {
    EXPECT_TRUE(std::any_of(first.val, first.val + kSize, [](bool v) { return v; }));
    EXPECT_TRUE(std::any_of(first.val, first.val + kSize, [](bool v) { return !v; }));
  } else {
    EXPECT_TRUE(
        std::all_of(first.val, first.val + kSize, [](NT v) { return v >= NT{-1} && v < NT{1}; }));
  }
}

}  // namespace spcraft::tests
