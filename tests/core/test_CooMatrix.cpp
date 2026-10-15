#include <gtest/gtest.h>

#include "SpCraft.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace spcraft::tests
{
namespace
{

template <class Entry>
bool EntriesEqual(const Entry& lhs, const Entry& rhs)
{
  return std::tie(lhs.row, lhs.col, lhs.val) == std::tie(rhs.row, rhs.col, rhs.val);
}

//! Entries deliberately out of row order, to prove ToCsr sorts by row.
spcraft::CooMatrix<std::int32_t, double, std::int64_t> Sample()
{
  spcraft::CooMatrix<std::int32_t, double, std::int64_t> matrix;
  matrix.Allocate(4, 3, 3);
  matrix.entries[0] = {2, 1, 40.0};
  matrix.entries[1] = {0, 0, 10.0};
  matrix.entries[2] = {1, 2, 30.0};
  matrix.entries[3] = {0, 2, 20.0};
  return matrix;
}

std::string WriteTemporaryMatrixMarket(const std::string& body)
{
  const std::string path = "spcraft_test_matrix.mtx";
  std::ofstream file(path);
  file << body;
  file.close();
  return path;
}

}  // namespace

TEST(CooMatrix, Correctness)
{
  using Matrix = spcraft::CooMatrix<std::int32_t, double, std::int64_t>;
  static_assert(std::is_same_v<Matrix::Entry, spcraft::TupleEntry<std::int32_t, double>>);
  static_assert(std::is_same_v<decltype(Matrix::entries), Matrix::Entry*>);
  static_assert(std::is_trivially_copyable_v<Matrix::Entry>);
  static_assert(std::is_trivially_destructible_v<Matrix::Entry>);
  static_assert(!std::is_copy_constructible_v<Matrix> && !std::is_copy_assignable_v<Matrix>);
  Matrix a;
  EXPECT_TRUE((!a.entries && a.nnz == 0 && a.m == 0 && a.n == 0 && a.memowned)) << "default";
  a.Allocate(2, 3, 4);
  EXPECT_TRUE((a.entries && a.memowned && a.nnz == 2 && a.m == 3 && a.n == 4)) << "allocate";
  a.entries[0] = {2, 1, 7};
  a.entries[1] = {0, 3, -5};
  const auto clone = a.Clone();
  EXPECT_TRUE((clone.entries != a.entries && EntriesEqual(clone.entries[1], a.entries[1]))) << "deep clone";
  auto* storage = a.entries;
  Matrix view(storage, a.nnz, a.m, a.n);
  EXPECT_TRUE((!view.memowned && view.entries == storage)) << "borrow";
  Matrix moved_view(std::move(view));
  EXPECT_TRUE((!view.entries && !moved_view.memowned && moved_view.entries == storage)) << "move view";
  moved_view.Reset();
  EXPECT_TRUE((!moved_view.entries && EntriesEqual(a.entries[0], clone.entries[0]))) << "reset view";
  Matrix moved(std::move(a));
  EXPECT_TRUE((!a.entries && !a.memowned && moved.entries == storage && moved.memowned)) << "move owner";
  a.Allocate(1, 1, 1);
  a = std::move(moved);
  EXPECT_TRUE((!moved.entries && a.entries == storage && a.memowned)) << "move assignment";
  auto& alias = a;
  a = std::move(alias);
  EXPECT_TRUE((a.entries == storage && EntriesEqual(a.entries[0], clone.entries[0]))) << "self move";
  bool threw = false;
  try {
    a.Allocate(std::numeric_limits<std::int64_t>::max(), 3, 4);
  } catch (const std::bad_array_new_length&) {
    threw = true;
  }
  EXPECT_TRUE((threw && a.entries == storage && a.nnz == 2)) << "failed allocation preserves owner";
  threw = false;
  try {
    a.Allocate(1, -1, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_TRUE((threw && a.entries == storage)) << "negative shape";
  a.Allocate(0, 3, 4);
  const auto empty = a.Clone();
  EXPECT_TRUE((!a.entries && a.nnz == 0 && a.memowned && empty.m == 3 && empty.n == 4 && !empty.entries))
      << "shaped empty allocation and clone";
  spcraft::CooMatrix<std::int32_t, bool> boolean;
  boolean.Allocate(1, 1, 1);
  ASSERT_NE(boolean.entries, nullptr);
  boolean.entries[0] = {0, 0, false};
  EXPECT_FALSE(boolean.entries[0].val) << "Boolean storage retains caller values";
  spcraft::CooMatrix<int, double> signed_count;
  threw = false;
  try {
    signed_count.Allocate(-1, 1, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_TRUE((threw)) << "negative count";
  const auto default_clone = Matrix{}.Clone();
  EXPECT_TRUE((!default_clone.entries && default_clone.m == 0 && default_clone.n == 0)) << "default clone";
  {
    // COO now stores only trivial entries; deterministic sizing failure tests cleanup.
    Matrix owned = Sample();
    auto* storage = owned.entries;
    constexpr auto kTooManyEntries = std::numeric_limits<std::int64_t>::max();
    EXPECT_THROW(owned.Allocate(kTooManyEntries, 2, 2), std::bad_array_new_length);
    EXPECT_EQ(owned.entries, storage);
    EXPECT_EQ(owned.nnz, 4);
    EXPECT_EQ(owned.m, 3);
    EXPECT_EQ(owned.n, 3);
  }

  // --- ToCsr groups entries by row, preserving values --------------------
  {
    const Matrix coo = Sample();
    const spcraft::CsrMatrix<std::int32_t, double, std::int64_t> csr = coo.ToCsr();
    EXPECT_TRUE((csr.m == 3 && csr.n == 3 && csr.nnz == 4)) << "ToCsr should preserve the shape and nnz";

    const std::int64_t expected_row_ptr[4] = {0, 2, 3, 4};
    EXPECT_TRUE((std::equal(expected_row_ptr, expected_row_ptr + 4, csr.row_ptr)))
        << "ToCsr should build the correct row pointers";
    // Row 0 holds columns 0 and 2 with values 10 and 20, in insertion order.
    EXPECT_TRUE((csr.col_id[0] == 0 && csr.val[0] == 10.0)) << "ToCsr row 0 first entry";
    EXPECT_TRUE((csr.col_id[1] == 2 && csr.val[1] == 20.0)) << "ToCsr row 0 second entry";
    EXPECT_TRUE((csr.col_id[2] == 2 && csr.val[2] == 30.0)) << "ToCsr row 1";
    EXPECT_TRUE((csr.col_id[3] == 1 && csr.val[3] == 40.0)) << "ToCsr row 2";
  }

  // --- ToCsr on an empty matrix ------------------------------------------
  {
    Matrix coo;
    coo.Allocate(0, 4, 4);
    bool threw = false;
    try {
      const auto csr = coo.ToCsr();
      EXPECT_TRUE((csr.m == 4 && csr.nnz == 0)) << "converting an empty matrix should give an empty CSR";
    } catch (const std::exception&) {
      threw = true;
    }
    EXPECT_TRUE((!threw)) << "converting an empty matrix must not throw";
  }

  // --- an out-of-range index is an error, not something to drop ----------
  {
    Matrix coo;
    coo.Allocate(2, 2, 2);
    coo.entries[0].row = 0;
    coo.entries[0].col = 0;
    coo.entries[0].val = 1.0;
    coo.entries[1].row = 7;  // outside a 2 x 2 matrix
    coo.entries[1].col = 0;
    coo.entries[1].val = 2.0;

    bool threw = false;
    try {
      const auto csr = coo.ToCsr();
      (void)csr;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    EXPECT_TRUE((threw)) << "ToCsr must reject an out-of-range index rather than silently drop it";
  }

  // --- Matrix Market round trip ------------------------------------------
  {
    const std::string path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 10.0\n"
        "2 3 20.0\n"
        "3 2 30.0\n");
    const Matrix coo = Matrix(path);
    EXPECT_TRUE((coo.m == 3 && coo.n == 3 && coo.nnz == 3)) << "Matrix Market input should read the header";
    EXPECT_TRUE((coo.entries[0].row == 0 && coo.entries[0].col == 0 && coo.entries[0].val == 10.0))
        << "Matrix Market input should convert to zero-based indices";
    std::remove(path.c_str());
  }

  // --- a structurally empty Matrix Market file is valid ------------------
  {
    const std::string path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 0\n");
    bool threw = false;
    try {
      const Matrix coo = Matrix(path);
      EXPECT_TRUE((coo.m == 3 && coo.n == 3 && coo.nnz == 0))
          << "an empty Matrix Market file should give an empty matrix";
    } catch (const std::exception&) {
      threw = true;
    }
    EXPECT_TRUE((!threw)) << "reading a structurally empty Matrix Market file must not throw";
    std::remove(path.c_str());
  }

  // --- a missing file is reported ----------------------------------------
  {
    bool threw = false;
    try {
      const Matrix coo = Matrix("this_file_does_not_exist.mtx");
      (void)coo;
    } catch (const std::runtime_error&) {
      threw = true;
    }
    EXPECT_TRUE((threw)) << "a missing Matrix Market file must throw";
  }

  // Arbitrary COO order, duplicates and explicit zeros survive conversion unchanged.
  {
    Matrix::Entry entries[] = {{2, 1, 0.0}, {0, 2, 4.0}, {0, 2, -4.0}, {2, 0, 7.0}};
    Matrix view(entries, 4, 3, 3);
    const auto csr = view.ToCsr();
    const std::int64_t offsets[] = {0, 2, 2, 4};
    EXPECT_TRUE((std::equal(offsets, offsets + 4, csr.row_ptr))) << "duplicate COO row offsets";
    EXPECT_TRUE((csr.nnz == 4 && csr.col_id[0] == 2 && csr.col_id[1] == 2 && csr.val[0] == 4.0 &&
                 csr.val[1] == -4.0 && csr.val[2] == 0.0 && csr.val[3] == 7.0))
        << "ToCsr preserves duplicates, zeros and within-row order";
    EXPECT_TRUE((EntriesEqual(entries[0], Matrix::Entry{2, 1, 0.0}))) << "ToCsr leaves borrowed input intact";
    const auto copy = view.Clone();
    EXPECT_TRUE((std::equal(entries, entries + 4, copy.entries, EntriesEqual<Matrix::Entry>)))
        << "Clone copies every component";
    view.Allocate(1, 1, 1);
    EXPECT_TRUE(
        (view.memowned && view.entries != entries && EntriesEqual(entries[3], Matrix::Entry{2, 0, 7.0})))
        << "Allocate replaces a view without freeing borrowed entries";
  }
  {
    spcraft::CooMatrix<std::int32_t, bool> coo;
    coo.Allocate(2, 2, 3);
    coo.entries[0] = {1, 2, true};
    coo.entries[1] = {0, 1, false};
    const auto csr = coo.ToCsr();
    EXPECT_TRUE((csr.row_ptr[2] == 2 && !csr.val[0] && csr.val[1])) << "Boolean ToCsr";
  }
  {
    Matrix matrix = Sample();
    auto* before = matrix.entries;
    bool invalid = false;
    try {
      matrix.Allocate(1, 0, 3);
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    EXPECT_TRUE((invalid && matrix.entries == before && matrix.nnz == 4))
        << "nonzero count with zero extent preserves owner on failure";
  }
  // Header dimensions must be checked before they narrow to the index type.
  {
    const Matrix::Entry invalid[] = {{-1, 0, 1}, {0, -1, 1}, {0, 3, 1}};
    for (const auto& entry : invalid) {
      Matrix coo;
      coo.Allocate(1, 2, 3);
      coo.entries[0] = entry;
      bool rejected = false;
      try {
        const auto csr = coo.ToCsr();
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      EXPECT_TRUE((rejected)) << "ToCsr rejects negative and out-of-range column coordinates";
    }
  }
  {
    const auto path =
        WriteTemporaryMatrixMarket("%%MatrixMarket matrix coordinate real general\n2147483648 1 0\n");
    bool overflow = false;
    try {
      const auto coo = Matrix(path);
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    EXPECT_TRUE((overflow)) << "Matrix Market dimensions beyond IT must throw overflow_error";
    std::remove(path.c_str());
  }
  {
    const auto path =
        WriteTemporaryMatrixMarket("%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n1 1\n3 1\n");
    const auto coo = Matrix(path);
    const auto csr = coo.ToCsr();
    EXPECT_TRUE((coo.nnz == 3 && csr.row_ptr[1] == 2 && csr.row_ptr[2] == 2 && csr.row_ptr[3] == 3))
        << "symmetric Matrix Market expansion retains one diagonal and both off-diagonals";
    EXPECT_TRUE((csr.val[0] == 1 && csr.val[1] == 1 && csr.val[2] == 1))
        << "pattern Matrix Market values remain one";
    std::remove(path.c_str());
  }
}

template <class Types>
class CooMatrixTypeTest : public ::testing::Test
{
};

using CooMatrixTypes = ::testing::Types<
    std::tuple<std::int32_t, float, std::int32_t>, std::tuple<std::int32_t, double, std::int32_t>,
    std::tuple<std::int32_t, bool, std::int32_t>, std::tuple<std::int32_t, float, std::int64_t>,
    std::tuple<std::int32_t, double, std::int64_t>, std::tuple<std::int32_t, bool, std::int64_t>,
    std::tuple<std::int64_t, float, std::int64_t>, std::tuple<std::int64_t, double, std::int64_t>,
    std::tuple<std::int64_t, bool, std::int64_t>>;
TYPED_TEST_SUITE(CooMatrixTypeTest, CooMatrixTypes);

TYPED_TEST(CooMatrixTypeTest, CompiledStorageAndConversions)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using TypedMatrix = spcraft::CooMatrix<IT, NT, OT>;
  TypedMatrix matrix;
  matrix.Allocate(3, 2, 3);
  ASSERT_NE(matrix.entries, nullptr);
  matrix.entries[0] = {1, 2, NT{1}};
  matrix.entries[1] = {0, 1, NT{0}};
  matrix.entries[2] = {0, 1, NT{1}};

  auto clone = matrix.Clone();
  ASSERT_NE(clone.entries, nullptr);
  EXPECT_NE(clone.entries, matrix.entries);
  clone.entries[0].val = NT{0};
  EXPECT_EQ(matrix.entries[0].val, NT{1});
  auto* storage = clone.entries;
  TypedMatrix moved(std::move(clone));
  EXPECT_EQ(moved.entries, storage);
  EXPECT_EQ(clone.entries, nullptr);
  EXPECT_FALSE(clone.memowned);
  TypedMatrix target;
  target.Allocate(1, 1, 1);
  target = std::move(moved);
  EXPECT_EQ(target.entries, storage);
  EXPECT_EQ(moved.entries, nullptr);
  auto& self = target;
  target = std::move(self);
  EXPECT_EQ(target.entries, storage);

  const auto csr = matrix.ToCsr();
  const auto csc = matrix.ToCsc();
  ASSERT_EQ(csr.nnz, OT{3});
  ASSERT_EQ(csc.nnz, OT{3});
  const OT row_offsets[] = {0, 2, 3};
  const OT col_offsets[] = {0, 0, 2, 3};
  EXPECT_TRUE(std::equal(row_offsets, row_offsets + 3, csr.row_ptr));
  EXPECT_TRUE(std::equal(col_offsets, col_offsets + 4, csc.col_ptr));
  EXPECT_EQ(csr.col_id[0], IT{1});
  EXPECT_EQ(csr.col_id[1], IT{1});
  EXPECT_EQ(csc.row_id[0], IT{0});
  EXPECT_EQ(csc.row_id[1], IT{0});
  EXPECT_EQ(csr.val[0], NT{0});
  EXPECT_EQ(csr.val[1], NT{1});
  EXPECT_EQ(csc.val[0], NT{0});
  EXPECT_EQ(csc.val[1], NT{1});

  target.Allocate(0, 2, 3);
  const auto empty = target.Clone();
  EXPECT_EQ(empty.entries, nullptr);
  EXPECT_EQ(empty.nnz, OT{0});
  EXPECT_EQ(empty.m, IT{2});
  EXPECT_EQ(empty.n, IT{3});
}

TYPED_TEST(CooMatrixTypeTest, CompiledMatrixMarketConstructor)
{
  using IT = std::tuple_element_t<0, TypeParam>;
  using NT = std::tuple_element_t<1, TypeParam>;
  using OT = std::tuple_element_t<2, TypeParam>;
  using TypedMatrix = spcraft::CooMatrix<IT, NT, OT>;
  const auto path =
      WriteTemporaryMatrixMarket("%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n1 1\n3 1\n");
  const TypedMatrix matrix(path);
  std::remove(path.c_str());
  EXPECT_EQ(matrix.m, IT{3});
  EXPECT_EQ(matrix.n, IT{3});
  ASSERT_EQ(matrix.nnz, OT{3});
  EXPECT_TRUE(std::all_of(matrix.entries, matrix.entries + matrix.nnz,
                          [](const auto& entry) { return entry.val == NT{1}; }));
  const auto csr = matrix.ToCsr();
  const OT offsets[] = {0, 2, 2, 3};
  EXPECT_TRUE(std::equal(offsets, offsets + 4, csr.row_ptr));
}

}  // namespace spcraft::tests
