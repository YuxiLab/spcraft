// Ownership and lifetime contract for DenseVector, plus the Random seeding it
// adds on top of the shared container pattern.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "SpCraft.h"

namespace
{

using Vector = spcraft::DenseVector<std::int32_t, double>;

int failures = 0;

void Check(bool condition, const char* what)
{
  if (!condition) {
    std::cerr << "DenseVector: " << what << "\n";
    ++failures;
  }
}

Vector Sample()
{
  Vector vector(4);
  for (std::int32_t index = 0; index < 4; ++index) {
    vector.val[index] = static_cast<double>(index) + 1.0;
  }
  return vector;
}

}  // namespace

int main()
{
  // --- default construction owns nothing ---------------------------------
  {
    Vector vector;
    Check(vector.val == nullptr && vector.n == 0,
          "a default-constructed vector should hold no buffer");
  }

  // --- the sizing constructor allocates ----------------------------------
  {
    Vector vector(6);
    Check(vector.n == 6 && vector.val != nullptr, "the sizing constructor should allocate");
    Check(vector.memowned, "the sizing constructor should take ownership");
  }

  // --- an empty vector is representable ----------------------------------
  {
    Vector vector(0);
    Check(vector.n == 0, "a zero-length vector should be allowed");
    Vector copy = vector.Clone();
    Check(copy.n == 0, "cloning an empty vector should work");
  }

  // --- Allocate replaces previous storage --------------------------------
  {
    Vector vector(4);
    vector.Allocate(9);
    Check(vector.n == 9 && vector.memowned, "a second Allocate should replace the first");
  }

  // --- the pointer constructor is the borrow path ------------------------
  {
    double buffer[3] = {1.0, 2.0, 3.0};
    {
      Vector view(buffer, 3);
      Check(!view.memowned, "the pointer constructor must not take ownership");
      Check(view.val == buffer, "a view should alias the caller's buffer");
    }
    Check(buffer[0] == 1.0, "a destroyed view must leave the caller's buffer alone");
  }

  // --- a non-empty view over a null pointer is rejected ------------------
  {
    bool threw = false;
    try {
      Vector view(nullptr, 5);
      (void)view;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "a non-empty view must reject a null buffer");
  }

  // --- move construction and assignment ----------------------------------
  {
    Vector source = Sample();
    const auto* buffer = source.val;
    Vector moved(std::move(source));
    Check(moved.val == buffer && moved.n == 4, "move construction should steal the buffer");
    Check(source.val == nullptr && source.n == 0 && !source.memowned,
          "a moved-from vector must release ownership");

    Vector target(10);
    target = std::move(moved);
    Check(target.val == buffer && target.n == 4, "move assignment should take the new storage");
    Check(moved.val == nullptr && !moved.memowned, "move assignment should empty the source");
  }

  // --- self-move assignment must not destroy the object ------------------
  {
    Vector vector = Sample();
    Vector& alias = vector;
    vector = std::move(alias);
    Check(vector.n == 4 && vector.val != nullptr, "self-move must leave the vector intact");
    Check(vector.val[0] == 1.0, "self-move must not corrupt the values");
  }

  // --- Clone is a deep copy ----------------------------------------------
  {
    Vector original = Sample();
    Vector copy = original.Clone();
    Check(copy.val != original.val, "Clone must not alias the source");
    Check(copy.n == original.n, "Clone should preserve the length");
    Check(std::equal(original.val, original.val + 4, copy.val), "Clone should copy the values");
    copy.val[0] = 99.0;
    Check(original.val[0] == 1.0, "writing to a clone must not touch the original");
  }

  // --- Reset releases without freeing ------------------------------------
  {
    Vector vector = Sample();
    auto* buffer = vector.val;
    vector.Release();
    Check(vector.val == nullptr && vector.n == 0 && !vector.memowned,
          "Reset should null every member and drop ownership");
    std::free(buffer);
  }

  // --- Random is reproducible and stays in range -------------------------
  {
    Vector first(64);
    Vector second(64);
    first.Random(1234);
    second.Random(1234);
    Check(std::equal(first.val, first.val + 64, second.val),
          "the same seed must produce the same values");
    Check(std::all_of(first.val, first.val + 64, [](double v) { return v >= -1.0 && v < 1.0; }),
          "real values must land in [-1, 1)");

    Vector third(64);
    third.Random(4321);
    Check(!std::equal(first.val, first.val + 64, third.val),
          "a different seed should produce different values");
  }

  // --- a request that cannot be sized is refused, not wrapped ------------
  {
    // size * sizeof(double) overflows std::size_t, so the multiplication would
    // wrap and hand back a buffer far smaller than requested.
    spcraft::DenseVector<std::int64_t, double> vector;
    bool threw = false;
    try {
      vector.Allocate(std::numeric_limits<std::int64_t>::max());
    } catch (const std::bad_array_new_length&) {
      threw = true;
    }
    Check(threw, "an unsizeable allocation must throw std::bad_array_new_length");
    Check(vector.n == 0 && vector.val == nullptr,
          "a failed Allocate must leave the vector untouched");
  }

  // --- negative sizes are rejected ---------------------------------------
  {
    Vector vector;
    bool threw = false;
    try {
      vector.Allocate(-1);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "a negative size must be rejected");
  }

  if (failures != 0) {
    std::cerr << failures << " DenseVector check(s) failed\n";
    return 1;
  }
  std::cout << "DenseVector checks passed\n";
  return 0;
}
