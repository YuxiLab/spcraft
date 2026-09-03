#include "cuSpMV.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

namespace
{

bool check_cuda(cudaError_t status, const char* operation)
{
  if (status == cudaSuccess) return true;
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

bool test_bfs_semiring()
{
  constexpr int rows = 3;
  constexpr std::array<int, rows + 1> row_ptr{0, 0, 1, 2};
  constexpr std::array<int, 2> col_id{0, 0};
  constexpr std::array<bool, 2> values{true, true};
  constexpr std::array<bool, rows> frontier{true, false, false};
  constexpr std::array<bool, rows> expected{false, true, true};
  std::array<bool, rows> result{};

  int* device_row_ptr = nullptr;
  int* device_col_id = nullptr;
  bool* device_values = nullptr;
  bool* device_frontier = nullptr;
  bool* device_result = nullptr;

  bool ok =
      check_cuda(cudaMalloc(&device_row_ptr, sizeof(row_ptr)), "cudaMalloc(BFS row_ptr)") &&
      check_cuda(cudaMalloc(&device_col_id, sizeof(col_id)), "cudaMalloc(BFS col_id)") &&
      check_cuda(cudaMalloc(&device_values, sizeof(values)), "cudaMalloc(BFS values)") &&
      check_cuda(cudaMalloc(&device_frontier, sizeof(frontier)), "cudaMalloc(BFS frontier)") &&
      check_cuda(cudaMalloc(&device_result, sizeof(result)), "cudaMalloc(BFS result)");

  if (ok) {
    ok = check_cuda(cudaMemcpy(device_row_ptr, row_ptr.data(), sizeof(row_ptr),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(BFS row_ptr)") &&
         check_cuda(
             cudaMemcpy(device_col_id, col_id.data(), sizeof(col_id), cudaMemcpyHostToDevice),
             "cudaMemcpy(BFS col_id)") &&
         check_cuda(
             cudaMemcpy(device_values, values.data(), sizeof(values), cudaMemcpyHostToDevice),
             "cudaMemcpy(BFS values)") &&
         check_cuda(cudaMemcpy(device_frontier, frontier.data(), sizeof(frontier),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(BFS frontier)");
  }

  if (ok) {
    spcraft::CsrMatrix<int, bool> device_matrix(device_row_ptr, device_col_id, device_values,
                                                static_cast<int>(values.size()), rows, rows);
    spcraft::DenseVector<int, bool> input(device_frontier, rows);
    spcraft::DenseVector<int, bool> output(device_result, rows);
    ok = check_cuda(
             spcraft::SpmvCudaBlockPerRow<spcraft::PlusTimesRing<bool>, int, bool, int, 32>(
                 device_matrix, input, output),
             "SpmvCudaBlockPerRow(BFS)") &&
         check_cuda(
             cudaMemcpy(result.data(), device_result, sizeof(result), cudaMemcpyDeviceToHost),
             "cudaMemcpy(BFS result)");
  }

  cudaFree(device_result);
  cudaFree(device_frontier);
  cudaFree(device_values);
  cudaFree(device_col_id);
  cudaFree(device_row_ptr);

  if (!ok) return false;
  if (result != expected) {
    std::cerr << "CUDA BFS semiring produced an incorrect frontier\n";
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  constexpr int rows = 4;
  constexpr std::array<int, rows + 1> row_ptr{0, 2, 4, 6, 7};
  constexpr std::array<int, 7> col_id{0, 3, 1, 4, 0, 2, 3};
  constexpr std::array<float, 7> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
  constexpr std::array<float, 5> x{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  constexpr std::array<float, rows> expected{9.0F, 26.0F, 23.0F, 28.0F};
  std::array<float, rows> result{};

  int* device_row_ptr = nullptr;
  int* device_col_id = nullptr;
  float* device_values = nullptr;
  float* device_x = nullptr;
  float* device_y = nullptr;

  bool ok = check_cuda(cudaMalloc(&device_row_ptr, sizeof(row_ptr)), "cudaMalloc(row_ptr)") &&
            check_cuda(cudaMalloc(&device_col_id, sizeof(col_id)), "cudaMalloc(col_id)") &&
            check_cuda(cudaMalloc(&device_values, sizeof(values)), "cudaMalloc(values)") &&
            check_cuda(cudaMalloc(&device_x, sizeof(x)), "cudaMalloc(x)") &&
            check_cuda(cudaMalloc(&device_y, sizeof(result)), "cudaMalloc(y)");

  if (ok) {
    ok = check_cuda(cudaMemcpy(device_row_ptr, row_ptr.data(), sizeof(row_ptr),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(row_ptr)") &&
         check_cuda(
             cudaMemcpy(device_col_id, col_id.data(), sizeof(col_id), cudaMemcpyHostToDevice),
             "cudaMemcpy(col_id)") &&
         check_cuda(
             cudaMemcpy(device_values, values.data(), sizeof(values), cudaMemcpyHostToDevice),
             "cudaMemcpy(values)") &&
         check_cuda(cudaMemcpy(device_x, x.data(), sizeof(x), cudaMemcpyHostToDevice),
                    "cudaMemcpy(x)");
  }

  if (ok) {
    spcraft::CsrMatrix<int, float> device_matrix(device_row_ptr, device_col_id, device_values,
                                                 values.size(), rows, x.size());
    spcraft::DenseVector<int, float> device_input(device_x, static_cast<int>(x.size()));
    spcraft::DenseVector<int, float> device_output(device_y, rows);
    spcraft::DenseVector<int, float> wrong_size_input(device_x,
                                                      static_cast<int>(x.size()) - 1);
    if (spcraft::SpmvCudaBlockPerRow<spcraft::PlusTimesRing<float>, int, float, int, 32>(
            device_matrix, wrong_size_input, device_output) != cudaErrorInvalidValue) {
      std::cerr << "CUDA SpMV accepted an input vector with the wrong size\n";
      ok = false;
    }
  }

  if (ok) {
    spcraft::CsrMatrix<int, float> device_matrix(device_row_ptr, device_col_id, device_values,
                                                 values.size(), rows, x.size());
    spcraft::DenseVector<int, float> device_input(device_x, static_cast<int>(x.size()));
    spcraft::DenseVector<int, float> device_output(device_y, rows);
    ok =
        check_cuda(
            spcraft::SpmvCudaBlockPerRow<spcraft::PlusTimesRing<float>, int, float, int, 32>(
                device_matrix, device_input, device_output),
            "SpmvCudaBlockPerRow") &&
        check_cuda(cudaMemcpy(result.data(), device_y, sizeof(result), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(y)");
  }

  cudaFree(device_y);
  cudaFree(device_x);
  cudaFree(device_values);
  cudaFree(device_col_id);
  cudaFree(device_row_ptr);

  if (!ok) return 1;
  for (int row = 0; row < rows; ++row) {
    if (std::abs(result[row] - expected[row]) > 1.0e-5F) {
      std::cerr << "incorrect result at row " << row << ": expected " << expected[row]
                << ", got " << result[row] << '\n';
      return 1;
    }
  }
  return test_bfs_semiring() ? 0 : 1;
}
