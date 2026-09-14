#include "SpCraft.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
int failures = 0;
void Check(bool ok, const char* message)
{
  if (!ok) {
    ++failures;
    std::cerr << message << '\n';
  }
}
using Matrix = spcraft::DcscMatrix<std::int32_t, double, std::int64_t>;
Matrix Sample()
{
  Matrix x;
  x.Allocate(3, 4, 9, 2);
  x.col_ptr[1] = 2;
  x.col_ptr[2] = 3;
  x.col_id[0] = 2;
  x.col_id[1] = 7;
  x.row_id[0] = 3;
  x.row_id[1] = 0;
  x.row_id[2] = 2;
  x.val[0] = 1;
  x.val[1] = 2;
  x.val[2] = 3;
  return x;
}
}  // namespace

int main()
{
  static_assert(!std::is_copy_constructible_v<Matrix> && !std::is_copy_assignable_v<Matrix>);
  static_assert(std::is_nothrow_move_constructible_v<Matrix>);
  Matrix empty;
  Check(empty.memowned && !empty.col_ptr && !empty.col_id && !empty.row_id && !empty.val &&
            empty.m == 0 && empty.n == 0 && empty.nnz == 0 && empty.nzc == 0,
        "default DCSC state");
  auto empty_clone = empty.Clone();
  Check(empty_clone.memowned && empty_clone.col_ptr && empty_clone.col_ptr[0] == 0,
        "default empty clone needs a terminal offset");
  empty.Allocate(0, 3, 100, 0);
  auto shaped_clone = empty.Clone();
  Check(shaped_clone.m == 3 && shaped_clone.n == 100 && shaped_clone.nzc == 0 &&
            !shaped_clone.col_id && shaped_clone.col_ptr[0] == 0,
        "shaped empty clone");
  {
    std::int64_t ptr[] = {0, 1};
    std::int32_t col[] = {4}, row[] = {1};
    double val[] = {7};
    Matrix view(ptr, col, row, val, 1, 2, 10, 1);
    Check(!view.memowned && view.col_id == col && view.val == val, "borrowed view aliases");
    Matrix moved(std::move(view));
    Check(!moved.memowned && moved.val == val && !view.val && !view.memowned,
          "moving a view preserves borrowing");
    moved.Allocate(0, 2, 10, 0);
    Check(val[0] == 7 && col[0] == 4 && moved.memowned,
          "Allocate on view leaves buffers alone");
  }
  {
    Matrix x = Sample();
    const auto* values = x.val;
    Matrix moved(std::move(x));
    Check(moved.val == values && moved.nzc == 2 && !x.val && !x.col_id && !x.memowned,
          "move construction transfers all buffers");
    Matrix target = Sample();
    target = std::move(moved);
    Check(target.val == values && !moved.val && moved.nzc == 0, "move assignment transfers");
    Matrix& alias = target;
    target = std::move(alias);
    Check(target.val == values && target.nnz == 3, "self-move retains data");
    auto clone = target.Clone();
    Check(clone.val != target.val && clone.col_id != target.col_id &&
              std::equal(clone.col_ptr, clone.col_ptr + 3, target.col_ptr) &&
              std::equal(clone.col_id, clone.col_id + 2, target.col_id) &&
              std::equal(clone.row_id, clone.row_id + 3, target.row_id) &&
              std::equal(clone.val, clone.val + 3, target.val),
          "deep clone copies every array");
    clone.val[0] = 99;
    Check(target.val[0] == 1, "clone mutation is independent");
    auto* p = clone.col_ptr;
    auto* c = clone.col_id;
    auto* r = clone.row_id;
    auto* v = clone.val;
    clone.Reset();
    Check(!clone.col_ptr && !clone.col_id && !clone.row_id && !clone.val && !clone.memowned &&
              clone.m == 0 && clone.n == 0 && clone.nzc == 0 && clone.nnz == 0,
          "Reset releases all members");
    std::free(p);
    std::free(c);
    std::free(r);
    std::free(v);
  }
  {
    Matrix x = Sample();
    const auto* old = x.val;
    bool threw = false;
    try {
      x.Allocate(std::numeric_limits<std::int64_t>::max(), 4, 9, 2);
    } catch (const std::bad_array_new_length&) {
      threw = true;
    }
    Check(threw && x.val == old && x.val[0] == 1 && x.nzc == 2,
          "unsizeable allocation preserves populated owner");
    for (int which = 0; which < 6; ++which) {
      threw = false;
      try {
        if (which == 0) x.Allocate(-1, 4, 9, 2);
        if (which == 1) x.Allocate(1, -1, 9, 1);
        if (which == 2) x.Allocate(1, 4, 9, -1);
        if (which == 3) x.Allocate(1, 4, 9, 2);
        if (which == 4) x.Allocate(0, 4, 9, 1);
        if (which == 5) x.Allocate(1, 4, 0, 1);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      Check(threw && x.val == old, "invalid counts must preserve storage");
    }
    auto csc = x.ToCsc();
    Check(csc.n == 9 && csc.col_ptr[2] == 0 && csc.col_ptr[3] == 2 && csc.col_ptr[7] == 2 &&
              csc.col_ptr[8] == 3 && csc.col_ptr[9] == 3,
          "CSC conversion restores empty-column offsets");
    auto roundtrip = Matrix::FromCsc(csc);
    Check(roundtrip.nzc == 2 && roundtrip.col_id[0] == 2 && roundtrip.col_id[1] == 7 &&
              std::equal(x.val, x.val + 3, roundtrip.val) &&
              std::equal(x.row_id, x.row_id + 3, roundtrip.row_id),
          "CSC roundtrip preserves unsorted rows and values");
    auto zero = Matrix::FromCsc(spcraft::CscMatrix<std::int32_t, double, std::int64_t>{});
    Check(zero.nnz == 0 && zero.nzc == 0, "default CSC conversion");
  }
  {
    // This logical width cannot have a dense offset array. DCSC needs two offsets only.
    spcraft::DcscMatrix<std::uint64_t, bool, std::uint64_t> huge;
    huge.Allocate(1, 1, std::numeric_limits<std::uint64_t>::max(), 1);
    huge.col_ptr[1] = 1;
    huge.col_id[0] = huge.n - 1;
    huge.val[0] = true;
    auto copy = huge.Clone();
    Check(copy.n == huge.n && copy.nzc == 1 && copy.val[0], "hypersparse unsigned clone");
  }
  if (failures)
    std::cerr << failures << " DCSC container checks failed\n";
  else
    std::cout << "DCSC container checks passed\n";
  return failures ? 1 : 0;
}
