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
    ok = check_cuda(cudaMemcpy(device_row_ptr, row_ptr.data(), sizeof(row_ptr), cudaMemcpyHostToDevice),
                    "cudaMemcpy(row_ptr)") &&
         check_cuda(cudaMemcpy(device_col_id, col_id.data(), sizeof(col_id), cudaMemcpyHostToDevice),
                    "cudaMemcpy(col_id)") &&
         check_cuda(cudaMemcpy(device_values, values.data(), sizeof(values), cudaMemcpyHostToDevice),
                    "cudaMemcpy(values)") &&
         check_cuda(cudaMemcpy(device_x, x.data(), sizeof(x), cudaMemcpyHostToDevice),
                    "cudaMemcpy(x)");
  }

  if (ok) {
    spcraft::CsrMatrix<int, float> device_matrix(
        device_row_ptr, device_col_id, device_values, values.size(), rows, x.size());
    ok = check_cuda(spcraft::spmv_cuda_block_per_row<
                        int, float, int, spcraft::SpmvBlockSize::B32>(
                        device_matrix, device_x, device_y),
                    "spmv_cuda_block_per_row") &&
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
  return 0;
}
