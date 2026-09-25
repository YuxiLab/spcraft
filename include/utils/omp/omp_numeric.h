#pragma once

#include "omp_wrapper.h"
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace spcraft
{
template <class IT, class OT>
void OmpPrefixSumImpl(const IT* in, OT* out, std::size_t size, int threads)
{
  std::vector<OT> tsum(threads + 1);
  OMP_PARALLEL(num_threads(threads))
  {
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    OT sum = 0;
    // First, sum each thread's own block.
    OMP_FOR(schedule(static))
    for (std::size_t i = 0; i < size; ++i) {
      sum += in[i];
      out[i + 1] = sum;
    }
    tsum[thread + 1] = sum;
    OMP_BARRIER
    OT offset = 0;
    // Then add the totals from earlier blocks to get global offsets.
    for (int i = 0; i < thread + 1; ++i) {
      offset += tsum[i];
    }
    OMP_FOR(schedule(static))
    for (std::size_t i = 0; i < size; ++i) {
      out[i + 1] += offset;
    }
  }
}

template <class IT, class OT = IT>
std::vector<OT> OmpPrefixSum(const std::vector<IT>& in, int threads)
{
  // Prefix inputs are nonnegative counts; validate before narrowing any partial sum.
  const auto total = std::accumulate(in.begin(), in.end(), int64_t{0});
  if (static_cast<std::uintmax_t>(total) >
      static_cast<std::uintmax_t>(std::numeric_limits<OT>::max())) {
    throw std::overflow_error("prefix sum exceeds output type");
  }
  const std::size_t size = in.size();
  std::vector<OT> out(size + 1);
  OmpPrefixSumImpl(in.data(), out.data(), size, threads);
  return out;
}

}  // namespace spcraft
