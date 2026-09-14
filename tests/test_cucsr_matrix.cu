// Ownership and lifetime contract for cuCsrMatrix: the same pattern as
// CsrMatrix, but over device memory, plus the host/device transfer paths.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "SpCraft.h"

namespace
{

using Host = spcraft::CsrMatrix<std::int32_t, double, std::int64_t>;
using Device = spcraft::cuCsrMatrix<std::int32_t, double, std::int64_t>;

int failures = 0;

void Check(bool condition, const char* what)
{
  if (!condition) {
    std::cerr << "cuCsrMatrix: " << what << "\n";
    ++failures;
  }
}

//! [ 1 0 2 ]
//! [ 0 3 0 ]
Host SampleHost()
{
  Host matrix;
  matrix.Allocate(3, 2, 3);
  matrix.row_ptr[0] = 0;
  matrix.row_ptr[1] = 2;
  matrix.row_ptr[2] = 3;
  matrix.col_id[0] = 0;
  matrix.col_id[1] = 2;
  matrix.col_id[2] = 1;
  matrix.val[0] = 1.0;
  matrix.val[1] = 2.0;
  matrix.val[2] = 3.0;
  return matrix;
}

bool SameAsSample(const Host& matrix)
{
  const std::int64_t row_ptr[3] = {0, 2, 3};
  const std::int32_t col_id[3] = {0, 2, 1};
  const double values[3] = {1.0, 2.0, 3.0};
  return matrix.m == 2 && matrix.n == 3 && matrix.nnz == 3 &&
         std::equal(row_ptr, row_ptr + 3, matrix.row_ptr) &&
         std::equal(col_id, col_id + 3, matrix.col_id) &&
         std::equal(values, values + 3, matrix.val);
}

}  // namespace

int main()
{
  int device_count = 0;
  const cudaError_t probe = cudaGetDeviceCount(&device_count);
  if (probe != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device available; skipping the cuCsrMatrix test\n";
    return 0;
  }

  // --- default construction owns nothing ---------------------------------
  {
    Device matrix;
    Check(matrix.row_ptr == nullptr && matrix.val == nullptr,
          "a default-constructed matrix should hold no buffers");
    Check(matrix.nnz == 0 && matrix.m == 0 && matrix.n == 0,
          "a default-constructed matrix should be empty");
  }

  // --- Allocate takes device storage and zeroes it -----------------------
  {
    Device matrix;
    matrix.Allocate(4, 3, 5);
    Check(matrix.nnz == 4 && matrix.m == 3 && matrix.n == 5,
          "Allocate should record the shape");
    Check(matrix.memowned, "Allocate should take ownership");

    // calloc semantics: the freshly allocated device buffers read back as zero.
    const Host downloaded = matrix.ToHost();
    Check(std::all_of(downloaded.val, downloaded.val + 4, [](double v) { return v == 0.0; }),
          "Allocate should zero the device values");
  }

  // --- an empty matrix is representable ----------------------------------
  {
    Device matrix;
    matrix.Allocate(0, 3, 3);
    Check(matrix.nnz == 0 && matrix.row_ptr != nullptr,
          "row_ptr is m+1 long even when nnz is zero");
    Device copy = matrix.Clone();
    Check(copy.nnz == 0 && copy.m == 3, "cloning an empty matrix should work");
  }

  // --- host round trip ---------------------------------------------------
  {
    const Host original = SampleHost();
    Device device = Device::FromHost(original);
    Check(device.m == 2 && device.n == 3 && device.nnz == 3,
          "FromHost should preserve the shape");
    Check(device.memowned, "FromHost should own its device storage");
    const Host back = device.ToHost();
    Check(SameAsSample(back), "a host round trip must preserve every entry");
  }

  // --- Clone is a device-to-device deep copy -----------------------------
  {
    Device device = Device::FromHost(SampleHost());
    Device copy = device.Clone();
    Check(copy.val != device.val, "Clone must not alias the source");
    Check(SameAsSample(copy.ToHost()), "Clone should copy every entry");
  }

  // --- View borrows without owning ---------------------------------------
  {
    Device device = Device::FromHost(SampleHost());
    {
      const Host view = device.View();
      Check(!view.memowned, "View must not take ownership");
      Check(view.row_ptr == device.row_ptr, "View should alias the device buffers");
      Check(view.m == 2 && view.nnz == 3, "View should carry the shape");
    }
    // The view is gone; the device matrix must still be intact.
    Check(SameAsSample(device.ToHost()), "destroying a view must not free device memory");
  }

  // --- the pointer constructor is the borrow path ------------------------
  {
    Device owner = Device::FromHost(SampleHost());
    {
      Device view(owner.row_ptr, owner.col_id, owner.val, owner.nnz, owner.m, owner.n);
      Check(!view.memowned, "the pointer constructor must not take ownership");
    }
    Check(SameAsSample(owner.ToHost()), "destroying a view must leave the owner intact");
  }

  // --- move construction and assignment ----------------------------------
  {
    Device source = Device::FromHost(SampleHost());
    const auto* buffer = source.val;
    Device moved(std::move(source));
    Check(moved.val == buffer && moved.nnz == 3, "move construction should steal the buffer");
    Check(source.val == nullptr && source.nnz == 0 && !source.memowned,
          "a moved-from matrix must release ownership");

    Device target;
    target.Allocate(9, 4, 4);
    target = std::move(moved);
    Check(target.val == buffer && target.nnz == 3,
          "move assignment should take the new storage");
    Check(moved.val == nullptr && !moved.memowned, "move assignment should empty the source");
    Check(SameAsSample(target.ToHost()), "the moved matrix should still hold its data");
  }

  // --- self-move assignment must not destroy the object ------------------
  {
    Device matrix = Device::FromHost(SampleHost());
    Device& alias = matrix;
    matrix = std::move(alias);
    Check(matrix.nnz == 3 && matrix.val != nullptr, "self-move must leave the matrix intact");
    Check(SameAsSample(matrix.ToHost()), "self-move must not corrupt the data");
  }

  // --- Reset releases without freeing ------------------------------------
  {
    Device matrix = Device::FromHost(SampleHost());
    auto* row_ptr = matrix.row_ptr;
    auto* col_id = matrix.col_id;
    auto* values = matrix.val;
    matrix.Reset();
    Check(matrix.row_ptr == nullptr && matrix.nnz == 0 && !matrix.memowned,
          "Reset should null every member and drop ownership");
    // Reset releases ownership without freeing, so the buffers are ours now.
    cudaFree(row_ptr);
    cudaFree(col_id);
    cudaFree(values);
  }

  // --- negative sizes are rejected ---------------------------------------
  {
    Device matrix;
    bool threw = false;
    try {
      matrix.Allocate(-1, 2, 2);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "a negative nnz must be rejected");
  }

  if (failures != 0) {
    std::cerr << failures << " cuCsrMatrix check(s) failed\n";
    return 1;
  }
  std::cout << "cuCsrMatrix checks passed\n";
  return 0;
}
