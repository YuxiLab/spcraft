#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "mpi/CommGrid2D.h"

namespace spcraft::detail
{

using DistGridPtr = std::shared_ptr<const CommGrid2D>;

int64_t LocalExtent(int64_t extent, int processes, int coordinate);

int64_t GlobalIndex(int64_t index, int64_t extent, int processes, int coordinate);

int64_t LocalIndex(int64_t index, int64_t extent, int processes, int coordinate);

//! Collective observation of nnz, independent of the local matrix's offset width.
int64_t GlobalNnz(const CommGrid2D& grid, int64_t local_nnz);

//! Validate global dimensions and compute this rank's contiguous local dimensions.
template <class IT>
std::pair<IT, IT> DistLocalDimensions(const DistGridPtr& grid, IT m, IT n)
{
  if (!grid || grid->World() == MPI_COMM_NULL) {
    throw std::invalid_argument("Distributed matrix needs a live communicator grid");
  }
  if constexpr (std::is_signed_v<IT>) {
    if (m < 0 || n < 0) {
      throw std::invalid_argument("Distributed matrix dimensions must be nonnegative");
    }
  }
  const IT rows = LocalExtent(m, grid->Rows(), grid->Row());
  const IT cols = LocalExtent(n, grid->Cols(), grid->Col());
  const std::pair<IT, IT> dimensions{rows, cols};
  return dimensions;
}

}  // namespace spcraft::detail
