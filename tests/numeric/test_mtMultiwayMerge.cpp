#include "SpCraft.h"

#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

using namespace spcraft;

using MergeTypes = testing::Types<std::tuple<int32_t, float, int32_t>, std::tuple<int32_t, double, int32_t>,
                                  std::tuple<int32_t, bool, int32_t>, std::tuple<int32_t, float, int64_t>,
                                  std::tuple<int32_t, double, int64_t>, std::tuple<int32_t, bool, int64_t>,
                                  std::tuple<int64_t, float, int64_t>, std::tuple<int64_t, double, int64_t>,
                                  std::tuple<int64_t, bool, int64_t>>;

template <class Types>
class MultiwayMergeTest : public testing::Test
{
};

TYPED_TEST_SUITE(MultiwayMergeTest, MergeTypes);

template <class IT, class NT, class OT>
SPCOO MakeMergeInput(std::initializer_list<TupleEntry<IT, NT>> entries)
{
  SPCOO matrix;
  matrix.Allocate((OT)entries.size(), IT{3}, IT{7});
  OT position = 0;
  for (const TupleEntry<IT, NT>& entry : entries) matrix.entries[position++] = entry;
  return matrix;
}

TYPED_TEST(MultiwayMergeTest, OverlappingCoordinatesKeepZerosAndPreserveInputs)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  std::vector<SPCOO> inputs;
  inputs.push_back(
      MakeMergeInput<IT, NT, OT>({{0, 0, NT{true}}, {0, 0, NT{true}}, {1, 2, NT{true}}, {0, 6, NT{false}}}));
  inputs.push_back(MakeMergeInput<IT, NT, OT>({{0, 0, NT{true}}, {2, 1, NT{false}}, {1, 2, NT{true}}}));
  inputs.push_back(MakeMergeInput<IT, NT, OT>({}));
  const TupleEntry<IT, NT>* original = inputs[0].entries;
  const IT expected_rows[] = {0, 2, 1, 0};
  const IT expected_cols[] = {0, 1, 2, 6};
  const NT expected_values[] = {(NT)3, NT{false}, (NT)2, NT{false}};

  const int original_threads = OMP_GET_MAX_THREADS();
  for (const int threads : {1, 2, 4}) {
    SCOPED_TRACE(testing::Message() << "threads=" << threads);
    OMP_SET_NUM_THREADS(threads);
    const SPCOO merged = OmpMultiwayMerge<PlusTimesRing<NT>>(inputs, IT{3}, IT{7});
    EXPECT_EQ(merged.m, 3);
    EXPECT_EQ(merged.n, 7);
    EXPECT_EQ(merged.nnz, 4);
    if (merged.nnz != 4) continue;
    for (OT position = 0; position < merged.nnz; ++position) {
      SCOPED_TRACE(testing::Message() << "entry=" << position);
      EXPECT_EQ(merged.entries[position].row, expected_rows[position]);
      EXPECT_EQ(merged.entries[position].col, expected_cols[position]);
      EXPECT_EQ(merged.entries[position].val, expected_values[position]);
    }
    EXPECT_NE(merged.entries, original);
    EXPECT_EQ(inputs[0].entries, original);
    EXPECT_EQ(inputs[0].nnz, 4);
    EXPECT_EQ(inputs[0].entries[0].val, NT{true});
    EXPECT_EQ(inputs[0].entries[1].val, NT{true});
  }
  OMP_SET_NUM_THREADS(original_threads);
}

TYPED_TEST(MultiwayMergeTest, SingleInputCombinesItsDuplicateCoordinates)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  std::vector<SPCOO> inputs;
  inputs.push_back(MakeMergeInput<IT, NT, OT>({{0, 0, NT{true}}, {0, 0, NT{true}}, {1, 2, NT{false}}}));
  const SPCOO merged = OmpMultiwayMerge<PlusTimesRing<NT>>(inputs, IT{3}, IT{7});
  ASSERT_EQ(merged.nnz, 2);
  EXPECT_EQ(merged.entries[0].row, 0);
  EXPECT_EQ(merged.entries[0].col, 0);
  EXPECT_EQ(merged.entries[0].val, (NT)2);
  EXPECT_EQ(merged.entries[1].row, 1);
  EXPECT_EQ(merged.entries[1].col, 2);
  EXPECT_EQ(merged.entries[1].val, NT{false});
  EXPECT_NE(merged.entries, inputs[0].entries);
}

TYPED_TEST(MultiwayMergeTest, EmptyInputsRetainRequestedShape)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  std::vector<SPCOO> inputs;
  for (const IT rows : {IT{0}, IT{3}}) {
    for (const IT cols : {IT{0}, IT{7}}) {
      const SPCOO merged = OmpMultiwayMerge<PlusTimesRing<NT>>(inputs, rows, cols);
      EXPECT_EQ(merged.m, rows);
      EXPECT_EQ(merged.n, cols);
      EXPECT_EQ(merged.nnz, 0);
    }
  }
  inputs.push_back(MakeMergeInput<IT, NT, OT>({}));
  inputs.push_back(MakeMergeInput<IT, NT, OT>({}));
  const SPCOO merged = OmpMultiwayMerge<PlusTimesRing<NT>>(inputs, IT{3}, IT{7});
  EXPECT_EQ(merged.m, 3);
  EXPECT_EQ(merged.n, 7);
  EXPECT_EQ(merged.nnz, 0);
}

TYPED_TEST(MultiwayMergeTest, RejectsMismatchedAndNegativeDimensions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using Ring = PlusTimesRing<NT>;
  std::vector<SPCOO> inputs;
  inputs.push_back(MakeMergeInput<IT, NT, OT>({}));
  EXPECT_THROW((void)OmpMultiwayMerge<Ring>(inputs, IT{3}, IT{8}), std::invalid_argument);
  EXPECT_THROW((void)OmpMultiwayMerge<Ring>(inputs, IT{4}, IT{7}), std::invalid_argument);
  EXPECT_THROW((void)OmpMultiwayMerge<Ring>(inputs, IT{-1}, IT{7}), std::invalid_argument);
  EXPECT_THROW((void)OmpMultiwayMerge<Ring>(inputs, IT{3}, IT{-1}), std::invalid_argument);
}
