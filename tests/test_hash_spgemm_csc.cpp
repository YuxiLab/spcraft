#include <gtest/gtest.h>

#include "SpCraft.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace spcraft::tests
{
namespace
{

template <class NT, class OT = int64_t, class IT = int32_t>
auto Make(IT rows, IT cols, const std::vector<spcraft::TupleEntry<IT, NT>>& entries)
{
  spcraft::CooMatrix<IT, NT, OT> coo;
  coo.Allocate(static_cast<OT>(entries.size()), rows, cols);
  for (std::size_t i = 0; i < entries.size(); ++i) coo.entries[i] = entries[i];
  return coo.ToCsc();
}

using Ring = spcraft::PlusTimesRing<double>;

struct MinPlus {
  using ValueType = double;
  static constexpr double kAdditiveIdentity = std::numeric_limits<double>::infinity();
  static double Add(double a, double b) { return std::min(a, b); }
  static double Multiply(double a, double b) { return a + b; }
};

void SmallChecks()
{
  // Includes an empty output column and two contributions to row zero.
  const auto a = Make<double>(2, 3, {{0, 0, 1}, {1, 1, 3}, {0, 2, 2}});
  const auto b = Make<double>(3, 3, {{0, 0, 4}, {2, 0, 6}, {1, 2, 5}, {2, 2, 7}});
  const auto c = spcraft::OmpHashSpGEMM<Ring>(a, b);
  EXPECT_EQ((c.m), (2)) << "hand product rows";
  EXPECT_EQ((c.n), (3)) << "hand product columns";
  EXPECT_EQ((c.nnz), (3)) << "hand product nnz";
  const spcraft::TupleEntry<int32_t, double> expected[] = {{0, 0, 16}, {0, 2, 14}, {1, 2, 15}};
  for (int i = 0; i < std::min<int64_t>(c.nnz, 3); ++i) {
    EXPECT_EQ((c.entries[i].row), (expected[i].row)) << "hand product row";
    EXPECT_EQ((c.entries[i].col), (expected[i].col)) << "hand product column";
    EXPECT_EQ((c.entries[i].val), (expected[i].val)) << "hand product value";
  }
  const auto empty = spcraft::OmpHashSpGEMM<Ring>(a, Make<double>(3, 4, {}));
  EXPECT_EQ((empty.nnz), (0)) << "empty product nnz";
  EXPECT_EQ((empty.m), (2)) << "empty product rows";
  EXPECT_EQ((empty.n), (4)) << "empty product columns";
  // Nonempty inputs can also have no structural paths through the inner dimension.
  const auto disjoint = spcraft::OmpHashSpGEMM<Ring>(Make<double>(1, 2, {{0, 0, 2}}),
                                                     Make<double>(2, 1, {{1, 0, 3}}));
  EXPECT_EQ((disjoint.nnz), (0)) << "disjoint product nnz";
  bool threw = false;
  try {
    const auto bad = spcraft::OmpHashSpGEMM<Ring>(a, Make<double>(4, 1, {}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_EQ((threw), (true)) << "dimension mismatch throws";

  const auto duplicate = spcraft::OmpHashSpGEMM<Ring>(Make<double>(1, 1, {{0, 0, 2}, {0, 0, -2}}),
                                                      Make<double>(1, 1, {{0, 0, 3}}));
  EXPECT_EQ((duplicate.nnz), (1)) << "cancellation retains structural entry";
  if (duplicate.nnz == 1) EXPECT_EQ((duplicate.entries[0].val), (0.0)) << "duplicate accumulation";
  const auto minplus = spcraft::OmpHashSpGEMM<MinPlus>(Make<double>(1, 2, {{0, 0, 2}, {0, 1, 5}}),
                                                       Make<double>(2, 1, {{0, 0, 7}, {1, 0, 1}}));
  EXPECT_EQ((minplus.nnz), (1)) << "min-plus nnz";
  if (minplus.nnz == 1) EXPECT_EQ((minplus.entries[0].val), (6.0)) << "min-plus value";
  const auto boolean = spcraft::OmpHashSpGEMM<spcraft::PlusTimesRing<bool>>(
      Make<bool>(1, 2, {{0, 0, true}, {0, 1, true}}),
      Make<bool>(2, 2, {{0, 0, false}, {1, 0, true}, {0, 1, false}}));
  EXPECT_EQ((boolean.nnz), (2)) << "Boolean structural nnz";
  if (boolean.nnz == 2) {
    EXPECT_EQ((boolean.entries[0].val), (true)) << "Boolean OR accumulation";
    EXPECT_EQ((boolean.entries[1].val), (false)) << "Boolean structural zero";
  }
}

template <class IT>
void CollisionChecks()
{
  // Sixteen keys with the same initial slot fill a sixteen-slot numeric table.
  std::vector<spcraft::TupleEntry<IT, double>> entries;
  for (IT i = 15; i >= 0; --i) entries.push_back({static_cast<IT>(i * 16), 0, double(i + 1)});
  const auto c =
      spcraft::OmpHashSpGEMM<Ring>(Make<double, int64_t, IT>(256, 1, entries),
                                   Make<double, int64_t, IT>(1, 1, {{0, 0, 2}, {0, 0, 3}}));
  EXPECT_EQ((c.nnz), (16)) << "collision product nnz";
  for (IT i = 0; i < std::min<int64_t>(c.nnz, 16); ++i) {
    EXPECT_EQ((c.entries[i].row), (i * 16)) << "collision product sorted row";
    EXPECT_EQ((c.entries[i].val), (double(5 * (i + 1)))) << "collision product value";
  }
}

void ReferenceChecks()
{
  // Dense multiplication and Boolean path enumeration are independent references.
  constexpr int rows = 7, inner = 9, cols = 5;
  for (int seed = 0; seed < 6; ++seed) {
    double a[rows][inner]{}, b[inner][cols]{};
    bool ap[rows][inner]{}, bp[inner][cols]{};
    std::vector<spcraft::TupleEntry<int32_t, double>> ae, be;
    for (int r = 0; r < rows; ++r)
      for (int k = 0; k < inner; ++k)
        if ((r + k + seed) % 3 == 0) {
          a[r][k] = (r * 2 + k) % 7 - 3;
          ap[r][k] = true;
          ae.push_back({r, k, a[r][k]});
        }
    for (int k = 0; k < inner; ++k)
      for (int col = 0; col < cols; ++col)
        if ((k * 2 + col + seed) % 4 == 0) {
          b[k][col] = (k + col) % 5 - 2;
          bp[k][col] = true;
          be.push_back({k, col, b[k][col]});
        }
    const auto c =
        spcraft::OmpHashSpGEMM<Ring>(Make<double>(rows, inner, ae), Make<double>(inner, cols, be));
    int pos = 0;
    for (int col = 0; col < cols; ++col)
      for (int row = 0; row < rows; ++row) {
        double value = 0;
        bool present = false;
        for (int k = 0; k < inner; ++k) {
          value += a[row][k] * b[k][col];
          present = present || (ap[row][k] && bp[k][col]);
        }
        if (!present) continue;
        if (pos < c.nnz) {
          EXPECT_EQ((c.entries[pos].row), (row)) << "reference row";
          EXPECT_EQ((c.entries[pos].col), (col)) << "reference column";
          EXPECT_EQ((c.entries[pos].val), (value)) << "reference value";
        }
        ++pos;
      }
    EXPECT_EQ((c.nnz), (pos)) << "reference structural nnz";
  }
}

void OverflowCheck()
{
  std::vector<spcraft::TupleEntry<int32_t, double>> ae, be;
  for (int i = 0; i < 16; ++i) {
    ae.push_back({i, 0, 1});
    be.push_back({0, i, 1});
  }
  bool threw = false;
  try {
    const auto c = spcraft::OmpHashSpGEMM<Ring>(Make<double, uint8_t>(16, 1, ae),
                                                Make<double, uint8_t>(1, 16, be));
  } catch (const std::overflow_error&) {
    threw = true;
  }
  EXPECT_EQ((threw), (true)) << "output nnz overflow throws before narrowing";
}
}  // namespace

class CscSpgemmTest : public ::testing::TestWithParam<int>
{
 protected:
  void SetUp() override { OMP_SET_NUM_THREADS(GetParam()); }
};
TEST_P(CscSpgemmTest, SemiringsAndEmptyColumns) { SmallChecks(); }
TEST_P(CscSpgemmTest, CollisionsWith32BitIndices) { CollisionChecks<std::int32_t>(); }
TEST_P(CscSpgemmTest, CollisionsWith64BitIndices) { CollisionChecks<std::int64_t>(); }
TEST_P(CscSpgemmTest, IndependentReference) { ReferenceChecks(); }
TEST_P(CscSpgemmTest, OutputOffsetOverflow) { OverflowCheck(); }
const std::vector<int> kThreadCounts =
    SPCRAFT_OPENMP_ENABLED ? std::vector<int>{1, 2, 4} : std::vector<int>{1};
INSTANTIATE_TEST_SUITE_P(OpenMPTeams, CscSpgemmTest, ::testing::ValuesIn(kThreadCounts));

}  // namespace spcraft::tests
