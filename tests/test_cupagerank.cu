#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "SpCraft.h"

namespace
{

using Index = std::int32_t;
using Value = double;
using Offset = std::int64_t;
using Matrix = spcraft::CsrMatrix<Index, Value, Offset>;
using DeviceMatrix = spcraft::cuCsrMatrix<Index, Value, Offset>;
using Vector = spcraft::DenseVector<Index, Value>;

bool CheckCuda(cudaError_t status, const char* what)
{
  if (status == cudaSuccess) return true;
  std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
  return false;
}

Matrix FromEdges(Index vertices, const std::vector<std::pair<Index, Index>>& edges)
{
  std::vector<Offset> row_ptr(static_cast<std::size_t>(vertices) + 1, 0);
  for (const auto& [from, to] : edges) {
    (void)to;
    ++row_ptr[static_cast<std::size_t>(from) + 1];
  }
  for (Index vertex = 0; vertex < vertices; ++vertex) {
    row_ptr[static_cast<std::size_t>(vertex) + 1] += row_ptr[static_cast<std::size_t>(vertex)];
  }

  Matrix matrix;
  matrix.Allocate(static_cast<Offset>(edges.size()), vertices, vertices);
  std::copy(row_ptr.begin(), row_ptr.end(), matrix.row_ptr);

  std::vector<Offset> cursor(row_ptr.begin(), row_ptr.end() - 1);
  for (const auto& [from, to] : edges) {
    const Offset destination = cursor[static_cast<std::size_t>(from)]++;
    matrix.col_id[destination] = to;
    matrix.val[destination] = 1.0;
  }
  return matrix;
}

//! Run both backends on the same operator and compare vertex by vertex.
int CompareBackends(const char* label, Index vertices,
                    const std::vector<std::pair<Index, Index>>& edges,
                    const spcraft::PageRankOptions<Value>& options)
{
  int failures = 0;
  const Matrix op = spcraft::pagerank_operator(FromEdges(vertices, edges));

  Vector host_rank(vertices);
  const auto host_result = spcraft::pagerank_openmp(op, host_rank, options);

  DeviceMatrix device_op = DeviceMatrix::FromHost(op);
  Value* device_values = nullptr;
  if (!CheckCuda(
          cudaMalloc(&device_values, static_cast<std::size_t>(vertices) * sizeof(Value)),
          "cudaMalloc(rank)")) {
    return 1;
  }
  Vector device_rank(device_values, vertices);

  spcraft::PageRankResult<Value> device_result;
  const cudaError_t status = spcraft::PageRankCuda<Index, Value, Offset, 128>(
      device_op.View(), device_rank, options, &device_result);
  if (!CheckCuda(status, "PageRankCuda")) ++failures;

  std::vector<Value> downloaded(static_cast<std::size_t>(vertices), 0.0);
  if (!CheckCuda(cudaMemcpy(downloaded.data(), device_values,
                            static_cast<std::size_t>(vertices) * sizeof(Value),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(rank)")) {
    ++failures;
  }
  cudaFree(device_values);

  if (device_result.converged != host_result.converged) {
    std::cerr << label << ": convergence disagrees between backends\n";
    ++failures;
  }
  for (Index vertex = 0; vertex < vertices; ++vertex) {
    const Value difference =
        std::abs(downloaded[static_cast<std::size_t>(vertex)] - host_rank.val[vertex]);
    if (difference > 1e-9) {
      std::cerr << label << ": rank[" << vertex << "] CUDA "
                << downloaded[static_cast<std::size_t>(vertex)] << " vs OpenMP "
                << host_rank.val[vertex] << "\n";
      ++failures;
    }
  }

  const Value mass = std::accumulate(downloaded.begin(), downloaded.end(), 0.0);
  if (std::abs(mass - 1.0) > 1e-12) {
    std::cerr << label << ": CUDA ranks sum to " << mass << ", expected 1\n";
    ++failures;
  }
  return failures;
}

}  // namespace

int main()
{
  int device_count = 0;
  const cudaError_t probe = cudaGetDeviceCount(&device_count);
  if (probe != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA device available; skipping the PageRank CUDA test\n";
    return 0;
  }

  int failures = 0;

  spcraft::PageRankOptions<Value> options;
  options.tolerance = 1e-14;
  options.max_iterations = 300;

  // A symmetric cycle, an asymmetric graph, and a graph with a dangling sink.
  failures += CompareBackends("3-cycle", 3, {{0, 1}, {1, 2}, {2, 0}}, options);
  failures += CompareBackends(
      "asymmetric", 6,
      {{0, 1}, {0, 2}, {1, 2}, {2, 0}, {3, 0}, {3, 1}, {3, 4}, {4, 5}, {5, 4}, {5, 0}},
      options);
  failures += CompareBackends("dangling sink", 3, {{0, 2}, {1, 2}}, options);

  // More vertices than one grid of blocks covers, so the grid-stride loops and
  // the two-stage reduction both take their multi-pass paths.
  {
    const Index vertices = 5000;
    std::vector<std::pair<Index, Index>> edges;
    for (Index vertex = 0; vertex < vertices; ++vertex) {
      edges.emplace_back(vertex, (vertex + 1) % vertices);
      if (vertex % 7 == 0) edges.emplace_back(vertex, (vertex * 3 + 11) % vertices);
    }
    spcraft::PageRankOptions<Value> large_options;
    large_options.tolerance = 1e-12;
    large_options.max_iterations = 200;
    failures += CompareBackends("large ring", vertices, edges, large_options);
  }

  // Shape and option validation must return an error, never throw.
  {
    Value* device_values = nullptr;
    if (CheckCuda(cudaMalloc(&device_values, 3 * sizeof(Value)), "cudaMalloc(validation)")) {
      const Matrix op = spcraft::pagerank_operator(FromEdges(3, {{0, 1}, {1, 2}, {2, 0}}));
      DeviceMatrix device_op = DeviceMatrix::FromHost(op);

      Matrix rectangular;
      rectangular.Allocate(0, 2, 3);
      DeviceMatrix device_rectangular = DeviceMatrix::FromHost(rectangular);
      Vector two(device_values, 2);
      if (spcraft::PageRankCuda(device_rectangular.View(), two) != cudaErrorInvalidValue) {
        std::cerr << "PageRankCuda accepted a rectangular operator\n";
        ++failures;
      }

      if (spcraft::PageRankCuda(device_op.View(), two) != cudaErrorInvalidValue) {
        std::cerr << "PageRankCuda accepted a rank vector of the wrong size\n";
        ++failures;
      }

      Vector rank(device_values, 3);
      spcraft::PageRankOptions<Value> bad;
      bad.damping = 1.0;
      if (spcraft::PageRankCuda(device_op.View(), rank, bad) != cudaErrorInvalidValue) {
        std::cerr << "PageRankCuda accepted a damping factor of 1\n";
        ++failures;
      }
      cudaFree(device_values);
    } else {
      ++failures;
    }
  }

  if (failures != 0) {
    std::cerr << failures << " PageRank CUDA check(s) failed\n";
    return 1;
  }
  std::cout << "PageRank CUDA checks passed\n";
  return 0;
}
