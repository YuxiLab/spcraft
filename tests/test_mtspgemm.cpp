#include "SpCraft.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#ifdef SPCRAFT_TEST_COMBBLAS
#include <CombBLAS/CombBLAS.h>
#include <mpi.h>
#endif

namespace
{
int failures = 0;
template <class A, class B>
void Equal(const A& a, const B& b, const char* label)
{
  if (a != b) {
    ++failures;
    std::cerr << label << ": got " << a << ", expected " << b << '\n';
  }
}
using Matrix = spcraft::DcscMatrix<int, double>;
using Ring = spcraft::PlusTimesRing<double>;

Matrix Make(int rows, int columns, const std::vector<std::tuple<int, int, double>>& entries)
{
  spcraft::CscMatrix<int, double> c;
  c.Allocate(entries.size(), rows, columns);
  auto sorted = entries;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return std::tie(std::get<1>(a), std::get<0>(a)) < std::tie(std::get<1>(b), std::get<0>(b));
  });
  for (std::size_t p = 0; p < sorted.size(); ++p) {
    c.row_id[p] = std::get<0>(sorted[p]);
    c.val[p] = std::get<2>(sorted[p]);
    ++c.col_ptr[std::get<1>(sorted[p]) + 1];
  }
  for (int col = 0; col < columns; ++col) c.col_ptr[col + 1] += c.col_ptr[col];
  return Matrix::FromCsc(c);
}

struct MinPlus {
  using ValueType = double;
  static constexpr double kAdditiveIdentity = std::numeric_limits<double>::infinity();
  static double Multiply(double a, double b) { return a + b; }
  static double Add(double a, double b) { return std::min(a, b); }
};

template <class IT, class NT, class OT>
void WidthChecks()
{
  spcraft::DcscMatrix<IT, NT, OT> a, b;
  a.Allocate(2, 2, 2, 2);
  b.Allocate(2, 2, 1, 1);
  a.col_ptr[1] = 1;
  a.col_ptr[2] = 2;
  a.col_id[1] = 1;
  a.row_id[1] = 1;
  a.val[0] = 2;
  a.val[1] = 3;
  b.col_ptr[1] = 2;
  b.row_id[1] = 1;
  b.val[0] = 5;
  b.val[1] = 7;
  const auto c = spcraft::OmpHashSpGEMM<spcraft::PlusTimesRing<NT>>(a, b);
  Equal(c.nnz, OT{2}, "width variant nnz");
  Equal(c.entries[0].val, NT{10}, "width variant first value");
  Equal(c.entries[1].val, NT{21}, "width variant second value");
}

void LocalChecks()
{
  auto a = Make(2, 3, {{0, 0, 1}, {1, 1, 3}, {0, 2, 2}});
  auto b = Make(3, 2, {{0, 0, 4}, {2, 0, 6}, {1, 1, 5}, {2, 1, 7}});
  const auto c = spcraft::OmpHashSpGEMM<Ring>(a, b);
  Equal(c.nnz, 3, "hand product nnz");
  const std::tuple<int, int, double> expected[] = {{0, 0, 16}, {0, 1, 14}, {1, 1, 15}};
  for (int i = 0; i < std::min(c.nnz, 3); ++i)
    Equal(std::tie(c.entries[i].row, c.entries[i].col, c.entries[i].val) == expected[i], true,
          "hand product tuple");
  static_assert(std::is_same_v<std::remove_cv_t<decltype(c)>, spcraft::CooMatrix<int, double>>);
  const auto csr = c.ToCsr();
  const int expected_offsets[] = {0, 2, 3};
  for (int i = 0; i < 3; ++i) {
    Equal(csr.row_ptr[i], expected_offsets[i], "native COO to CSR offsets");
    Equal(csr.col_id[i], std::get<1>(expected[i]), "native COO to CSR column");
    Equal(csr.val[i], std::get<2>(expected[i]), "native COO to CSR value");
  }
  Matrix empty;
  empty.Allocate(0, 3, 7, 0);
  const auto zero = spcraft::OmpHashSpGEMM<Ring>(a, empty);
  Equal(zero.nnz, 0, "empty product count");
  Equal(zero.m, 2, "empty product rows");
  Equal(zero.n, 7, "empty product columns");
  bool threw = false;
  try {
    const auto bad = spcraft::OmpHashSpGEMM<Ring>(b, empty);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Equal(threw, true, "dimension mismatch");
  a = Make(1, 1, {{0, 0, -0.0}});
  b = Make(1, 1, {{0, 0, 1}});
  Equal(spcraft::OmpHashSpGEMM<Ring>(a, b).entries[0].val, 0.0, "zero product value");
  a = Make(1, 2, {{0, 0, 2}, {0, 1, 5}});
  b = Make(2, 1, {{0, 0, 7}, {1, 0, 1}});
  Equal(spcraft::OmpHashSpGEMM<MinPlus>(a, b).entries[0].val, 6.0, "min-plus");
  spcraft::DcscMatrix<int, bool> ba, bb;
  ba.Allocate(1, 1, 1, 1);
  bb.Allocate(1, 1, 1, 1);
  ba.col_ptr[1] = bb.col_ptr[1] = 1;
  ba.val[0] = true;
  bb.val[0] = false;
  const auto bc = spcraft::OmpHashSpGEMM<spcraft::PlusTimesRing<bool>>(ba, bb);
  Equal(bc.nnz, 1, "Boolean structural zero nnz");
  Equal(bc.entries[0].val, false, "Boolean product");
  const std::vector<std::uint8_t> exact{100, 100, 55}, excess{100, 100, 56};
  Equal(+spcraft::OmpPrefixSum(exact, 4).back(), 255, "exact maximum prefix");
  threw = false;
  try {
    const auto p = spcraft::OmpPrefixSum(excess, 4);
  } catch (const std::overflow_error&) {
    threw = true;
  }
  Equal(threw, true, "prefix overflow");
}

template <class Storage>
concept AcceptsHashProduct = requires(const Storage& a) { spcraft::OmpHashSpGEMM<Ring>(a, a); };
static_assert(AcceptsHashProduct<Matrix>);
static_assert(AcceptsHashProduct<spcraft::CsrMatrix<int, double>>);
static_assert(AcceptsHashProduct<spcraft::CscMatrix<int, double>>);

void CompressedColumnChecks()
{
  const auto a = Make(4, 20, {{3, 2, 2}, {0, 2, 1}, {0, 2, -1}, {1, 17, 4}});
  const auto b = Make(20, 100, {{2, 7, 3}, {5, 7, 9}, {17, 91, 2}});
  const auto c = spcraft::OmpHashSpGEMM<Ring>(a, b);
  Equal(c.m, 4, "compressed product rows");
  Equal(c.n, 100, "compressed product logical width");
  Equal(c.nnz, 3, "compressed product nnz");
  const spcraft::TupleEntry<int, double> expected[] = {{0, 7, 0}, {3, 7, 6}, {1, 91, 8}};
  for (int i = 0; i < std::min(c.nnz, 3); ++i) {
    Equal(c.entries[i].row, expected[i].row, "compressed product row");
    Equal(c.entries[i].col, expected[i].col, "compressed product logical column");
    Equal(c.entries[i].val, expected[i].val, "compressed product value");
  }
  const auto disjoint = spcraft::OmpHashSpGEMM<Ring>(a, Make(20, 100, {{5, 91, 1}}));
  Equal(disjoint.nnz, 0, "disjoint stored columns");
  Equal(disjoint.n, 100, "disjoint product shape");

  std::vector<std::tuple<int, int, double>> entries;
  for (int i = 15; i >= 0; --i) entries.emplace_back(16 * i, 2, i + 1);
  const auto collision =
      spcraft::OmpHashSpGEMM<Ring>(Make(256, 20, entries), Make(20, 100, {{2, 91, 5}}));
  Equal(collision.nnz, 16, "DCSC collision count");
  for (int i = 0; i < std::min(collision.nnz, 16); ++i) {
    Equal(collision.entries[i].row, 16 * i, "DCSC collision sorted row");
    Equal(collision.entries[i].col, 91, "DCSC collision logical column");
    Equal(collision.entries[i].val, 5.0 * (i + 1), "DCSC collision value");
  }
}

void SparseLookupChecks()
{
  // These logical dimensions cannot be expanded to dense column pointers.
  using IT = int64_t;
  using OT = int64_t;
  constexpr IT width = (IT{1} << 40) + 37;
  using WideMatrix = spcraft::DcscMatrix<IT, double, OT>;
  WideMatrix a;
  a.Allocate(6, 3, width, 6);
  const IT columns[] = {0, 1, 2, width / 2, width - 2, width - 1};
  for (IT col = 0; col < 6; ++col) {
    a.col_id[col] = columns[col];
    a.col_ptr[col + 1] = col + 1;
    a.row_id[col] = col % 3;
    a.val[col] = col + 1;
  }
  const std::vector<std::vector<IT>> requests = {
      {0},
      {width - 1},
      {width / 3},
      {width / 2 - 1},
      {0, 0, 1, 2, width / 3, width / 2, width - 2, width - 1},
      {width - 1, 0, width / 3, width - 2, 0, width / 2, 1, 2}};
  for (const auto& rows : requests) {
    WideMatrix b;
    b.Allocate(rows.size(), width, width, 1);
    b.col_id[0] = width - 3;
    b.col_ptr[1] = rows.size();
    double expected[3]{};
    bool present[3]{};
    for (std::size_t p = 0; p < rows.size(); ++p) {
      b.row_id[p] = rows[p];
      b.val[p] = p + 1;
      // Independent direct search avoids sharing the chunk/merge logic.
      for (IT col = 0; col < a.nzc; ++col) {
        if (rows[p] == a.col_id[col]) {
          expected[a.row_id[col]] += a.val[col] * b.val[p];
          present[a.row_id[col]] = true;
        }
      }
    }
    const auto c = spcraft::OmpHashSpGEMM<Ring>(a, b);
    Equal(c.m, IT{3}, "wide product rows");
    Equal(c.n, width, "wide product columns");
    OT pos = 0;
    for (IT row = 0; row < 3; ++row) {
      if (!present[row]) continue;
      if (pos < c.nnz) {
        Equal(c.entries[pos].row, row, "wide lookup output row");
        Equal(c.entries[pos].col, width - 3, "wide lookup output column");
        Equal(c.entries[pos].val, expected[row], "wide lookup output value");
      }
      ++pos;
    }
    Equal(c.nnz, pos, "wide lookup output count");
  }

  // OR accumulation needs more than a single Boolean contribution.
  spcraft::DcscMatrix<int, bool> ba, bb;
  ba.Allocate(2, 1, 3, 2);
  ba.col_id[1] = 2;
  ba.col_ptr[1] = 1;
  ba.col_ptr[2] = 2;
  ba.val[0] = ba.val[1] = true;
  bb.Allocate(3, 3, 10, 2);
  bb.col_id[0] = 3;
  bb.col_id[1] = 9;
  bb.col_ptr[1] = 2;
  bb.col_ptr[2] = 3;
  bb.row_id[1] = 2;
  bb.val[1] = true;
  const auto bc = spcraft::OmpHashSpGEMM<spcraft::PlusTimesRing<bool>>(ba, bb);
  Equal(bc.nnz, 2, "Boolean accumulated structural count");
  if (bc.nnz == 2) {
    Equal(bc.entries[0].val, true, "Boolean OR across contributions");
    Equal(bc.entries[1].val, false, "Boolean accumulated structural zero");
  }
}

void IndependentChecks()
{
  // Dense arithmetic and a separate Boolean reachability pattern provide a
  // reference independent of hash tables, column lookup and output prefixes.
  for (int threads : {1, 2, 4}) {
#if SPCRAFT_OPENMP_ENABLED
    OMP_SET_NUM_THREADS(threads);
#else
    if (threads != 1) continue;
#endif
    CompressedColumnChecks();
    SparseLookupChecks();
    for (int seed = 0; seed < 12; ++seed) {
      constexpr int rows = 7, inner = 19, columns = 11;
      std::vector<std::tuple<int, int, double>> ae, be;
      double a[rows][inner]{}, b[inner][columns]{};
      bool ap[rows][inner]{}, bp[inner][columns]{};
      for (int r = 0; r < rows; ++r)
        for (int k = 0; k < inner; ++k)
          if ((r * 7 + k * 3 + seed) % 5 == 0) {
            a[r][k] = (r + k + seed) % 7 - 3;
            ap[r][k] = true;
            ae.emplace_back(r, k, a[r][k]);
          }
      for (int k = 0; k < inner; ++k)
        for (int c = 0; c < columns; ++c)
          if ((k * 3 + c + seed) % 4 == 0) {
            b[k][c] = (k + c) % 5 - 2;
            bp[k][c] = true;
            be.emplace_back(k, c, b[k][c]);
          }
      const auto result =
          spcraft::OmpHashSpGEMM<Ring>(Make(rows, inner, ae), Make(inner, columns, be));
      int p = 0;
      for (int c = 0; c < columns; ++c)
        for (int r = 0; r < rows; ++r) {
          bool present = false;
          double value = 0;
          for (int k = 0; k < inner; ++k) {
            present = present || (ap[r][k] && bp[k][c]);
            value += a[r][k] * b[k][c];
          }
          if (present) {
            if (p < result.nnz) {
              Equal(result.entries[p].row, r, "dense-reference row");
              Equal(result.entries[p].col, c, "dense-reference column");
              Equal(result.entries[p].val, value, "dense-reference value");
            }
            ++p;
          }
        }
      Equal(result.nnz, p, "dense-reference structural nnz");
    }
  }
}

struct RelationRing {
  using ValueType = unsigned;
  static constexpr unsigned kAdditiveIdentity = 0;
  static unsigned Add(unsigned a, unsigned b) { return a | b; }
  static unsigned Multiply(unsigned a, unsigned b)
  {
    unsigned result = 0;
    constexpr int order = 2;
    for (int r = 0; r < order; ++r)
      for (int c = 0; c < order; ++c)
        for (int k = 0; k < order; ++k)
          if ((a & (1U << (r * order + k))) && (b & (1U << (k * order + c))))
            result |= 1U << (r * order + c);
    return result;
  }
};
void BoundaryChecks()
{
  spcraft::DcscMatrix<int, unsigned> forward, backward;
  forward.Allocate(1, 1, 1, 1);
  backward.Allocate(1, 1, 1, 1);
  forward.col_ptr[1] = backward.col_ptr[1] = 1;
  forward.val[0] = 2;
  backward.val[0] = 4;
  const auto composed = spcraft::OmpHashSpGEMM<RelationRing>(forward, backward);
  Equal(composed.nnz, 1, "noncommutative product nnz");
  Equal(composed.entries[0].val, 1U, "noncommutative multiplication order");

  using Narrow = spcraft::DcscMatrix<int, double, std::uint8_t>;
  Narrow a, b;
  constexpr int rows = 15, inner = 17;
  a.Allocate(rows * inner, rows, inner, inner);
  b.Allocate(inner, inner, inner, inner);
  for (int k = 0; k < inner; ++k) {
    a.col_id[k] = b.col_id[k] = k;
    a.col_ptr[k + 1] = (k + 1) * rows;
    b.col_ptr[k + 1] = k + 1;
    b.row_id[k] = k;
    b.val[k] = 1;
    for (int r = 0; r < rows; ++r) {
      a.row_id[k * rows + r] = r;
      a.val[k * rows + r] = 2;
    }
  }
  const auto exact = spcraft::OmpHashSpGEMM<Ring>(a, b);
  Equal(+exact.nnz, 255, "maximum representable output nnz");
  Equal(exact.entries[254].val, 2.0, "last entry at maximum output nnz");

  a.Allocate(32, 2, 16, 16);
  b.Allocate(128, 16, 8, 8);
  for (int k = 0; k < 16; ++k) {
    a.col_id[k] = k;
    a.col_ptr[k + 1] = 2 * (k + 1);
    a.row_id[2 * k + 1] = 1;
  }
  for (int c = 0; c < 8; ++c) {
    b.col_id[c] = c;
    b.col_ptr[c + 1] = 16 * (c + 1);
    for (int k = 0; k < 16; ++k) b.row_id[16 * c + k] = k;
  }
  // Work may exceed OT even when the output fits; work prefixes use int64_t.
  const auto many_paths = spcraft::OmpHashSpGEMM<Ring>(a, b);
  Equal(+many_paths.nnz, 16, "work exceeding offset range with representable output");

  a.Allocate(32, 32, 1, 1);
  b.Allocate(8, 1, 8, 8);
  a.col_ptr[1] = 32;
  for (int row = 0; row < 32; ++row) a.row_id[row] = row;
  for (int col = 0; col < 8; ++col) {
    b.col_id[col] = col;
    b.col_ptr[col + 1] = col + 1;
  }
  bool overflow = false;
  try {
    const auto excess = spcraft::OmpHashSpGEMM<Ring>(a, b);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  Equal(overflow, true, "output total overflow rejected before narrowing");
}

#ifdef SPCRAFT_TEST_COMBBLAS
using Comb = combblas::SpDCCols<int, double>;
using CombRing = combblas::PlusTimesSRing<double, double>;
Comb Convert(const Matrix& a)
{
  combblas::SpTuples<int, double> t(a.nnz, a.m, a.n);
  for (int col = 0; col < a.nzc; ++col)
    for (int p = a.col_ptr[col]; p < a.col_ptr[col + 1]; ++p) {
      t.rowindex(p) = a.row_id[p];
      t.colindex(p) = a.col_id[col];
      t.numvalue(p) = a.val[p];
    }
  return Comb(t, false);
}
void Compare(const Matrix& a, const Matrix& b)
{
  const auto ca = Convert(a), cb = Convert(b);
  const auto x = spcraft::OmpHashSpGEMM<Ring>(a, b);
  const std::unique_ptr<combblas::SpTuples<int, double>> y(
      combblas::LocalSpGEMMHash<CombRing, double>(ca, cb, false, false, true));
  Equal(x.nnz, y->getnnz(), "native tuple count");
  for (int i = 0; i < std::min(x.nnz, static_cast<int>(y->getnnz())); ++i) {
    Equal(x.entries[i].row, y->rowindex(i), "tuple row");
    Equal(x.entries[i].col, y->colindex(i), "tuple column");
    Equal(x.entries[i].val, y->numvalue(i), "reference value");
  }
}
void ReferenceChecks()
{
  for (int threads : {1, 2, 4}) {
#if SPCRAFT_OPENMP_ENABLED
    OMP_SET_NUM_THREADS(threads);
#else
    if (threads != 1) continue;
#endif
    for (int seed = 0; seed < 12; ++seed) {
      std::vector<std::tuple<int, int, double>> ae, be;
      for (int col = 0; col < 32; ++col)
        for (int row = 0; row < 9; ++row)
          if ((row * 7 + col * 3 + seed) % 5 == 0)
            ae.emplace_back(row, col, double((row + col) % 7 - 3));
      for (int col = 0; col < 19; ++col)
        for (int row = 0; row < 32; ++row)
          if ((row + seed) % ((col % 3 == 0) ? 2 : 17) == 0)
            be.emplace_back(row, col, double((row + col) % 5 - 2));
      Compare(Make(9, 32, ae), Make(32, 19, be));
    }
    std::vector<std::tuple<int, int, double>> ae, be;
    for (int col = 0; col < 32; ++col) {
      ae.emplace_back(0, col, 1);
      ae.emplace_back(1, col, 1);
      be.emplace_back(col, 0, 1);
    }
    Compare(Make(2, 32, ae), Make(32, 1, be));
    Compare(Make(2, 100, {{0, 1, 2}, {1, 99, 3}}),
            Make(100, 3, {{50, 0, 4}, {1, 2, 5}, {99, 2, 6}}));
    ae.clear();
    for (int i = 1; i < 10; ++i) ae.emplace_back(i % 2, i, i);
    ae.emplace_back(1, 99, 3);
    Compare(Make(2, 100, ae), Make(100, 3, {{50, 0, 4}, {1, 2, 5}, {99, 2, 6}}));
    Compare(Make(1, 1, {{0, 0, -0.0}}), Make(1, 1, {{0, 0, 1}}));
  }
}
#endif
}  // namespace
int main(int argc, char** argv)
{
#ifdef SPCRAFT_TEST_COMBBLAS
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided < MPI_THREAD_FUNNELED) {
    std::cerr << "MPI thread support: got " << provided << ", expected FUNNELED\n";
    MPI_Finalize();
    return 1;
  }
#else
  (void)argc;
  (void)argv;
#endif
  OMP_SET_DYNAMIC(0);
  OMP_SET_NUM_THREADS(4);
  LocalChecks();
  IndependentChecks();
  BoundaryChecks();
  WidthChecks<std::int32_t, float, std::int32_t>();
  WidthChecks<std::uint32_t, double, std::uint64_t>();
  WidthChecks<std::int64_t, double, std::int64_t>();
  WidthChecks<std::int32_t, double, std::int64_t>();
#ifdef SPCRAFT_TEST_COMBBLAS
  ReferenceChecks();
  MPI_Finalize();
#endif
  if (!failures) std::cout << "OmpHashSpGEMM checks passed\n";
  return failures ? 1 : 0;
}
