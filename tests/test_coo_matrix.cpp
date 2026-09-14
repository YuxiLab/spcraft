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

namespace
{
int failures = 0;
struct Tracked {
  static inline int live = 0;
  static inline int remaining = -1;
  Tracked()
  {
    if (remaining == 0) throw std::runtime_error("injected value construction failure");
    if (remaining > 0) --remaining;
    ++live;
  }
  ~Tracked() { --live; }
};
void Check(bool ok, const char* message)
{
  if (!ok) {
    ++failures;
    std::cerr << message << ": got false, expected true\n";
  }
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
int main()
{
  using Matrix = spcraft::CooMatrix<std::int32_t, double, std::int64_t>;
  static_assert(std::is_same_v<Matrix::Entry, std::tuple<std::int32_t, std::int32_t, double>>);
  static_assert(std::is_same_v<decltype(Matrix::entries), Matrix::Entry*>);
  static_assert(!std::is_copy_constructible_v<Matrix> && !std::is_copy_assignable_v<Matrix>);
  Matrix a;
  Check(!a.entries && a.nnz == 0 && a.m == 0 && a.n == 0 && a.memowned, "default");
  a.Allocate(2, 3, 4);
  Check(a.entries && a.memowned && a.nnz == 2 && a.m == 3 && a.n == 4, "allocate");
  Check(a.entries[0] == Matrix::Entry{0, 0, 0}, "tuple initialization");
  a.entries[0] = {2, 1, 7};
  a.entries[1] = {0, 3, -5};
  const auto clone = a.Clone();
  Check(clone.entries != a.entries && clone.entries[1] == a.entries[1], "deep clone");
  auto* storage = a.entries;
  Matrix view(storage, a.nnz, a.m, a.n);
  Check(!view.memowned && view.entries == storage, "borrow");
  Matrix moved_view(std::move(view));
  Check(!view.entries && !moved_view.memowned && moved_view.entries == storage, "move view");
  moved_view.Release();
  Check(!moved_view.entries && a.entries[0] == clone.entries[0], "reset view");
  Matrix moved(std::move(a));
  Check(!a.entries && !a.memowned && moved.entries == storage && moved.memowned, "move owner");
  a.Allocate(1, 1, 1);
  a = std::move(moved);
  Check(!moved.entries && a.entries == storage && a.memowned, "move assignment");
  auto& alias = a;
  a = std::move(alias);
  Check(a.entries == storage && a.entries[0] == clone.entries[0], "self move");
  bool threw = false;
  try {
    a.Allocate(std::numeric_limits<std::int64_t>::max(), 3, 4);
  } catch (const std::bad_array_new_length&) {
    threw = true;
  }
  Check(threw && a.entries == storage && a.nnz == 2, "failed allocation preserves owner");
  threw = false;
  try {
    a.Allocate(1, -1, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw && a.entries == storage, "negative shape");
  a.Allocate(0, 3, 4);
  const auto empty = a.Clone();
  Check(!a.entries && a.nnz == 0 && a.memowned && empty.m == 3 && empty.n == 4 && !empty.entries,
        "shaped empty allocation and clone");
  spcraft::CooMatrix<unsigned, bool> boolean;
  boolean.Allocate(1, 1, 1);
  Check(!std::get<2>(boolean.entries[0]), "unsigned Boolean allocation");
  spcraft::CooMatrix<int, double> signed_count;
  threw = false;
  try {
    signed_count.Allocate(-1, 1, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "negative count");
  const auto default_clone = Matrix{}.Clone();
  Check(!default_clone.entries && default_clone.m == 0 && default_clone.n == 0, "default clone");
  {
    spcraft::CooMatrix<int, Tracked> objects;
    objects.Allocate(2, 2, 2);
    auto* old = objects.entries;
    Tracked::remaining = 2;
    threw = false;
    try {
      objects.Allocate(4, 2, 2);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw && Tracked::live == 2 && objects.entries == old && objects.nnz == 2,
          "partial tuple construction cleanup preserves existing owner");
    Tracked::remaining = -1;
  }
  Check(Tracked::live == 0, "nontrivial tuple destruction");
  // --- ToCsr groups entries by row, preserving values --------------------
  {
    const Matrix coo = Sample();
    const spcraft::CsrMatrix<std::int32_t, double, std::int64_t> csr = coo.ToCsr();
    Check(csr.m == 3 && csr.n == 3 && csr.nnz == 4, "ToCsr should preserve the shape and nnz");

    const std::int64_t expected_row_ptr[4] = {0, 2, 3, 4};
    Check(std::equal(expected_row_ptr, expected_row_ptr + 4, csr.row_ptr),
          "ToCsr should build the correct row pointers");
    // Row 0 holds columns 0 and 2 with values 10 and 20, in insertion order.
    Check(csr.col_id[0] == 0 && csr.val[0] == 10.0, "ToCsr row 0 first entry");
    Check(csr.col_id[1] == 2 && csr.val[1] == 20.0, "ToCsr row 0 second entry");
    Check(csr.col_id[2] == 2 && csr.val[2] == 30.0, "ToCsr row 1");
    Check(csr.col_id[3] == 1 && csr.val[3] == 40.0, "ToCsr row 2");
  }

  // --- ToCsr on an empty matrix ------------------------------------------
  {
    Matrix coo;
    coo.Allocate(0, 4, 4);
    bool threw = false;
    try {
      const auto csr = coo.ToCsr();
      Check(csr.m == 4 && csr.nnz == 0, "converting an empty matrix should give an empty CSR");
    } catch (const std::exception&) {
      threw = true;
    }
    Check(!threw, "converting an empty matrix must not throw");
  }

  // --- an out-of-range index is an error, not something to drop ----------
  {
    Matrix coo;
    coo.Allocate(2, 2, 2);
    std::get<0>(coo.entries[0]) = 0;
    std::get<1>(coo.entries[0]) = 0;
    std::get<2>(coo.entries[0]) = 1.0;
    std::get<0>(coo.entries[1]) = 7;  // outside a 2 x 2 matrix
    std::get<1>(coo.entries[1]) = 0;
    std::get<2>(coo.entries[1]) = 2.0;

    bool threw = false;
    try {
      const auto csr = coo.ToCsr();
      (void)csr;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "ToCsr must reject an out-of-range index rather than silently drop it");
  }

  // --- Matrix Market round trip ------------------------------------------
  {
    const std::string path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 10.0\n"
        "2 3 20.0\n"
        "3 2 30.0\n");
    const Matrix coo = Matrix::FromMatrixMarket(path);
    Check(coo.m == 3 && coo.n == 3 && coo.nnz == 3, "FromMatrixMarket should read the header");
    Check(std::get<0>(coo.entries[0]) == 0 && std::get<1>(coo.entries[0]) == 0 &&
              std::get<2>(coo.entries[0]) == 10.0,
          "FromMatrixMarket should convert to zero-based indices");
    std::remove(path.c_str());
  }

  // --- a structurally empty Matrix Market file is valid ------------------
  {
    const std::string path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 0\n");
    bool threw = false;
    try {
      const Matrix coo = Matrix::FromMatrixMarket(path);
      Check(coo.m == 3 && coo.n == 3 && coo.nnz == 0,
            "an empty Matrix Market file should give an empty matrix");
    } catch (const std::exception&) {
      threw = true;
    }
    Check(!threw, "reading a structurally empty Matrix Market file must not throw");
    std::remove(path.c_str());
  }

  // --- a missing file is reported ----------------------------------------
  {
    bool threw = false;
    try {
      const Matrix coo = Matrix::FromMatrixMarket("this_file_does_not_exist.mtx");
      (void)coo;
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "a missing Matrix Market file must throw");
  }

  // Arbitrary COO order, duplicates and explicit zeros survive conversion unchanged.
  {
    Matrix::Entry entries[] = {{2, 1, 0.0}, {0, 2, 4.0}, {0, 2, -4.0}, {2, 0, 7.0}};
    Matrix view(entries, 4, 3, 3);
    const auto csr = view.ToCsr();
    const std::int64_t offsets[] = {0, 2, 2, 4};
    Check(std::equal(offsets, offsets + 4, csr.row_ptr), "duplicate COO row offsets");
    Check(csr.nnz == 4 && csr.col_id[0] == 2 && csr.col_id[1] == 2 && csr.val[0] == 4.0 &&
              csr.val[1] == -4.0 && csr.val[2] == 0.0 && csr.val[3] == 7.0,
          "ToCsr preserves duplicates, zeros and within-row order");
    Check(entries[0] == Matrix::Entry{2, 1, 0.0}, "ToCsr leaves borrowed input intact");
    const auto copy = view.Clone();
    Check(std::equal(entries, entries + 4, copy.entries), "Clone copies every component");
    view.Allocate(1, 1, 1);
    Check(view.memowned && view.entries != entries && entries[3] == Matrix::Entry{2, 0, 7.0},
          "Allocate replaces a view without freeing borrowed entries");
  }
  {
    spcraft::CooMatrix<unsigned, bool> coo;
    coo.Allocate(2, 2, 3);
    coo.entries[0] = {1, 2, true};
    coo.entries[1] = {0, 1, false};
    const auto csr = coo.ToCsr();
    Check(csr.row_ptr[2] == 2 && !csr.val[0] && csr.val[1], "unsigned Boolean ToCsr");
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
    Check(invalid && matrix.entries == before && matrix.nnz == 4,
          "nonzero count with zero extent preserves owner on failure");
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
      Check(rejected, "ToCsr rejects negative and out-of-range column coordinates");
    }
  }
  {
    const auto path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate real general\n2147483648 1 0\n");
    bool overflow = false;
    try {
      const auto coo = Matrix::FromMatrixMarket(path);
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    Check(overflow, "Matrix Market dimensions beyond IT must throw overflow_error");
    std::remove(path.c_str());
  }
  {
    const auto path = WriteTemporaryMatrixMarket(
        "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n1 1\n3 1\n");
    const auto coo = Matrix::FromMatrixMarket(path);
    const auto csr = coo.ToCsr();
    Check(coo.nnz == 3 && csr.row_ptr[1] == 2 && csr.row_ptr[2] == 2 && csr.row_ptr[3] == 3,
          "symmetric Matrix Market expansion retains one diagonal and both off-diagonals");
    Check(csr.val[0] == 1 && csr.val[1] == 1 && csr.val[2] == 1,
          "pattern Matrix Market values remain one");
    std::remove(path.c_str());
  }
  if (!failures) std::cout << "CooMatrix checks passed\n";
  return failures ? 1 : 0;
}
