#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "SpCraft.h"

namespace spcraft::tests
{

template <class Matrix>
auto CudaMatrixData(const Matrix& matrix)
{
  if constexpr (requires { matrix.entries; }) {
    return matrix.entries;
  } else {
    return matrix.val;
  }
}

template <class Matrix>
bool SameMatrix(const Matrix& actual, const Matrix& expected)
{
  if (actual.m != expected.m || actual.n != expected.n || actual.nnz != expected.nnz) return false;
  if constexpr (requires { actual.entries; }) {
    for (decltype(actual.nnz) p = 0; p < actual.nnz; ++p) {
      if (actual.entries[p].row != expected.entries[p].row ||
          actual.entries[p].col != expected.entries[p].col ||
          actual.entries[p].val != expected.entries[p].val) {
        return false;
      }
    }
  } else {
    if constexpr (requires { actual.row_ptr; }) {
      if (!std::equal(actual.row_ptr, actual.row_ptr + actual.m + 1, expected.row_ptr))
        return false;
      if (actual.nnz && !std::equal(actual.col_id, actual.col_id + actual.nnz, expected.col_id)) {
        return false;
      }
    } else {
      if (!std::equal(actual.col_ptr, actual.col_ptr + actual.n + 1, expected.col_ptr))
        return false;
      if (actual.nnz && !std::equal(actual.row_id, actual.row_id + actual.nnz, expected.row_id)) {
        return false;
      }
    }
    if (actual.nnz && !std::equal(actual.val, actual.val + actual.nnz, expected.val)) return false;
  }
  return true;
}

template <class IT, class NT, class OT, class Host, class Device>
int CheckCudaContainer(const char* name)
{
  static_assert(!std::is_copy_constructible_v<Device> && !std::is_copy_assignable_v<Device>);
  static_assert(std::is_nothrow_move_constructible_v<Device>);
  static_assert(std::is_nothrow_move_assignable_v<Device>);
  int failures = 0;
  const auto check = [&](bool condition, const char* operation) {
    if (!condition) {
      std::cerr << name << " (IT=" << sizeof(IT) * 8 << ", NT="
                << (std::is_same_v<NT, bool>    ? "bool"
                    : std::is_same_v<NT, float> ? "float"
                                                : "double")
                << ", OT=" << sizeof(OT) * 8 << "): " << operation << '\n';
      ++failures;
    }
  };

  // A rectangular shape detects accidentally using m+1 for CSC column pointers.
  constexpr IT rows = 3;
  constexpr IT columns = 5;
  constexpr OT count = 3;
  Host expected;
  expected.Allocate(count, rows, columns);
  if constexpr (requires { expected.entries; }) {
    expected.entries[0] = {2, 4, NT{1}};
    expected.entries[1] = {0, 1, NT{0}};
    expected.entries[2] = {1, 0, NT{1}};
  } else {
    if constexpr (requires { expected.row_ptr; }) {
      const OT pointers[] = {0, 2, 2, 3};
      const IT indices[] = {0, 4, 1};
      std::copy_n(pointers, rows + 1, expected.row_ptr);
      std::copy_n(indices, count, expected.col_id);
    } else {
      const OT pointers[] = {0, 2, 2, 2, 2, 3};
      const IT indices[] = {0, 2, 1};
      std::copy_n(pointers, columns + 1, expected.col_ptr);
      std::copy_n(indices, count, expected.row_id);
    }
    const NT values[] = {NT{1}, NT{0}, NT{1}};
    std::copy_n(values, count, expected.val);
  }

  // Default and shaped empty matrices remain valid through upload and cloning.
  {
    Device empty;
    check(empty.nnz == 0 && empty.m == 0 && empty.n == 0 && !CudaMatrixData(empty),
          "default construction must be empty");
    const auto default_host = empty.ToHost();
    check(default_host.nnz == 0 && default_host.m == 0 && default_host.n == 0,
          "default download must preserve the zero shape");
    if constexpr (requires { default_host.row_ptr; }) {
      check(default_host.row_ptr[0] == 0, "default CSR download must initialize its pointer");
    } else if constexpr (requires { default_host.col_ptr; }) {
      check(default_host.col_ptr[0] == 0, "default CSC download must initialize its pointer");
    }
    check(empty.Clone().nnz == 0, "default clone must be empty");
    empty.Allocate(0, rows, columns);
    const auto copy = empty.Clone();
    const auto downloaded = copy.ToHost();
    check(copy.memowned && copy.nnz == 0 && copy.m == rows && copy.n == columns,
          "empty clone must retain its shape and ownership");
    if constexpr (requires { downloaded.row_ptr; }) {
      const auto zero = [](OT value) { return value == 0; };
      check(std::all_of(downloaded.row_ptr, downloaded.row_ptr + rows + 1, zero),
            "empty CSR pointers must be zero");
    } else if constexpr (requires { downloaded.col_ptr; }) {
      const auto zero = [](OT value) { return value == 0; };
      check(std::all_of(downloaded.col_ptr, downloaded.col_ptr + columns + 1, zero),
            "empty CSC pointers must be zero");
    }
    const Device uploaded(downloaded);
    check(SameMatrix(uploaded.ToHost(), downloaded), "empty upload must preserve storage");
  }

  // Round trips and deep copies preserve values and actual device provenance.
  Device device(expected);
  check(device.memowned && SameMatrix(device.ToHost(), expected), "upload/download must match");
  cudaPointerAttributes attributes{};
  const auto status = cudaPointerGetAttributes(&attributes, CudaMatrixData(device));
  check(status == cudaSuccess && attributes.type == cudaMemoryTypeDevice,
        "storage must reside in device memory");
  Device copied = device.Clone();
  check(CudaMatrixData(copied) != CudaMatrixData(device), "Clone must allocate separate storage");
  check(SameMatrix(copied.ToHost(), expected), "Clone must preserve every entry");
  {
    auto independent = device.ToHost();
    if constexpr (requires { independent.entries; }) {
      independent.entries[0].val = NT{0};
    } else {
      independent.val[0] = NT{0};
    }
    check(SameMatrix(device.ToHost(), expected), "downloads must not alias device storage");
    const auto view = device.View();
    check(!view.memowned && CudaMatrixData(view) == CudaMatrixData(device),
          "View must borrow the same device storage");
  }
  {
    Device borrowed;
    if constexpr (requires { device.entries; }) {
      borrowed = Device(device.entries, device.nnz, device.m, device.n);
    } else if constexpr (requires { device.row_ptr; }) {
      borrowed = Device(device.row_ptr, device.col_id, device.val, device.nnz, device.m, device.n);
    } else {
      borrowed = Device(device.col_ptr, device.row_id, device.val, device.nnz, device.m, device.n);
    }
    check(!borrowed.memowned && SameMatrix(borrowed.ToHost(), expected),
          "pointer construction must borrow device storage");
  }
  check(SameMatrix(device.ToHost(), expected), "destroying views must leave the owner intact");

  // Moves transfer ownership; self-move and failed replacement preserve the data.
  const auto buffer = CudaMatrixData(device);
  Device moved(std::move(device));
  check(CudaMatrixData(moved) == buffer && !CudaMatrixData(device) && !device.memowned,
        "move construction must steal storage and reset the source");
  device.Allocate(count, rows, columns);
  const auto zeroed = device.ToHost();
  if constexpr (requires { zeroed.entries; }) {
    for (OT p = 0; p < count; ++p) {
      check(zeroed.entries[p].row == 0 && zeroed.entries[p].col == 0 &&
                zeroed.entries[p].val == NT{0},
            "Allocate must zero every entry");
    }
  } else {
    const auto zero = [](NT value) { return value == NT{0}; };
    check(std::all_of(zeroed.val, zeroed.val + count, zero), "Allocate must zero values");
  }
  device = std::move(moved);
  check(CudaMatrixData(device) == buffer && !CudaMatrixData(moved) && !moved.memowned,
        "move assignment must replace storage and reset the source");
  Device& alias = device;
  device = std::move(alias);
  check(SameMatrix(device.ToHost(), expected), "self-move must retain the data");
  bool rejected = false;
  try {
    device.Allocate(OT{-1}, rows, columns);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected && CudaMatrixData(device) == buffer && SameMatrix(device.ToHost(), expected),
        "negative count must throw and preserve the previous allocation");
  if constexpr (sizeof(OT) == sizeof(std::int64_t)) {
    rejected = false;
    try {
      device.Allocate(std::numeric_limits<OT>::max(), rows, columns);
    } catch (const std::bad_array_new_length&) {
      rejected = true;
    }
    check(rejected && CudaMatrixData(device) == buffer && SameMatrix(device.ToHost(), expected),
          "an unsizeable allocation must throw before CUDA and preserve the old data");
  }
  rejected = false;
  try {
    device.Allocate(count, 0, columns);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected && SameMatrix(device.ToHost(), expected), "nonempty zero-row shape must throw");
  {
    Host missing;
    missing.nnz = count;
    missing.m = rows;
    missing.n = columns;
    rejected = false;
    try {
      const Device invalid(missing);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "upload with missing host storage must throw without leaking device storage");
  }

  // Reset deliberately releases storage without freeing it; the caller frees it.
  auto released = copied.View();
  copied.Reset();
  check(!copied.memowned && copied.nnz == 0 && copied.m == 0 && copied.n == 0 &&
            !CudaMatrixData(copied),
        "Reset must empty the object and release ownership");
  cudaPointerAttributes released_attributes{};
  check(cudaPointerGetAttributes(&released_attributes, CudaMatrixData(released)) == cudaSuccess,
        "Reset must leave the released allocation alive");
  if constexpr (requires { released.entries; }) {
    check(cudaFree(released.entries) == cudaSuccess, "released COO entries must be freeable");
  } else {
    if constexpr (requires { released.row_ptr; }) {
      check(cudaFree(released.row_ptr) == cudaSuccess, "released CSR pointers must be freeable");
      check(cudaFree(released.col_id) == cudaSuccess, "released CSR indices must be freeable");
    } else {
      check(cudaFree(released.col_ptr) == cudaSuccess, "released CSC pointers must be freeable");
      check(cudaFree(released.row_id) == cudaSuccess, "released CSC indices must be freeable");
    }
    check(cudaFree(released.val) == cudaSuccess, "released values must be freeable");
  }
  return failures;
}

template <template <class, class, class> class Host, template <class, class, class> class Device>
int CheckCudaContainerTypes(const char* name)
{
  int failures = 0;
  const auto check_values = [&]<class IT, class OT>() {
    failures += CheckCudaContainer<IT, float, OT, Host<IT, float, OT>, Device<IT, float, OT>>(name);
    failures +=
        CheckCudaContainer<IT, double, OT, Host<IT, double, OT>, Device<IT, double, OT>>(name);
    failures += CheckCudaContainer<IT, bool, OT, Host<IT, bool, OT>, Device<IT, bool, OT>>(name);
  };
  check_values.template operator()<std::int32_t, std::int32_t>();
  check_values.template operator()<std::int32_t, std::int64_t>();
  check_values.template operator()<std::int64_t, std::int64_t>();
  std::cout << name << ": checked all nine type combinations; " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}

}  // namespace spcraft::tests
