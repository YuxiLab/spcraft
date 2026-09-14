#pragma once

#ifdef SPCRAFT_USE_CUDA

#include <cuda_runtime.h>

namespace spcraft
{
//! Turn a failed CUDA status into an exception; out-of-memory keeps std::bad_alloc.
void CUDACHK(cudaError_t status, const char* what);
}  // namespace spcraft

#endif  // SPCRAFT_USE_CUDA
