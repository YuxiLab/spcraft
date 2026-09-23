#pragma once

#include "omp_wrapper.h"

template <class OT>
std::vector<OT> OmpPrefixSum(const std::vector<OT>& in, int threads)
{
  const auto size = in.size();
  std::vector<OT> out(size + 1);
  std::vector<OT> tsum(threads + 1);
  int overflow = 0;
  OMP_PARALLEL(num_threads(threads) default(none) shared(in, out, tsum, size, overflow))
  {
    int thread = 0;
    thread = OMP_GET_THREAD_NUM();
    OT sum = 0;
    int local_overflow = 0;
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
    if (local_overflow) {
      OMP_CRITICAL
      overflow = 1;
    }
  }
  if (overflow) {
    throw std::overflow_error("CombBLAS-mapped prefix sum exceeds offset type limit");
  }
  return out;
}