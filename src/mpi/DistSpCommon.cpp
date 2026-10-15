#include "mpi/DistSpCommon.h"

#include <algorithm>

namespace spcraft::detail
{

int64_t LocalExtent(int64_t extent, int processes, int coordinate)
{
  const int64_t block = extent / processes + (extent % processes != 0);
  const int64_t begin = coordinate * block;
  return std::max(int64_t{0}, std::min(block, extent - begin));
}

int64_t GlobalIndex(int64_t index, int64_t extent, int processes, int coordinate)
{
  const int64_t block = extent / processes + (extent % processes != 0);
  return coordinate * block + index;
}

int64_t LocalIndex(int64_t index, int64_t extent, int processes, int coordinate)
{
  if (index < 0 || index >= extent) throw std::out_of_range("Global index is outside the matrix");
  const int64_t block = extent / processes + (extent % processes != 0);
  if (index / block != coordinate) {
    throw std::out_of_range("Global index belongs to another process coordinate");
  }
  return index - coordinate * block;
}

int64_t GlobalNnz(const CommGrid2D& grid, int64_t local_nnz)
{
  int64_t global_nnz = 0;
  MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, grid.World());
  return global_nnz;
}

}  // namespace spcraft::detail
