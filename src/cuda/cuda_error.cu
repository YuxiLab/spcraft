#include <stdexcept>
#include <string>
#include <fmt/format.h>
#include "utils/cuda/cuda_error.h"

namespace spcraft
{
void CUDACHK(cudaError_t status, const char* what)
{
  if (status == cudaSuccess) return;
  if (status == cudaErrorMemoryAllocation) {
    throw std::bad_alloc();
  }
  throw std::runtime_error(fmt::format("{}: {}", what, cudaGetErrorString(status)));
}

}  // namespace spcraft
