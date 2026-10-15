#include "SpCraft.h"
#include "CudaMatrixTestCommon.h"

namespace
{

int TestRun()
{
  return spcraft::tests::CheckCudaContainerTypes<spcraft::CooMatrix, spcraft::cuCooMatrix>("cuCooMatrix");
}

}  // namespace

int main()
{
  int device_count = 0;
  const auto probe = cudaGetDeviceCount(&device_count);
  if (probe != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device available; skipping the cuCooMatrix test\n";
    return 0;
  }
  try {
    const int result = TestRun();
    const auto status = cudaDeviceReset();
    return result != 0 || status != cudaSuccess ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "cuCooMatrix: " << error.what() << '\n';
    return 1;
  }
}
