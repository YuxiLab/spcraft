#pragma once

#include <Eigen/SparseCore>
#include <cxxopts.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "SpCraft.h"

namespace spcraft::benchmarks
{

//! Input matrix plus the label used in report rows.
template <class IT, class NT, class OT>
struct MatrixInput {
  std::shared_ptr<spcraft::CsrMatrix<IT, NT, OT>> matrix;
  std::string name;
};

/**
 * @brief Options every SPCraft kernel benchmark accepts.
 *
 * The index and offset defaults are arguments because a backend's representable
 * set is its own: oneMKL's CSR is MKL_INT on both, so its driver defaults the
 * offset to 32 bits rather than leaving a default that selects nothing.
 */
inline void AddCommonOptions(cxxopts::Options& options, const char* default_index = "int32",
                             const char* default_offset = "int64")
{
  // clang-format off
  options.add_options()
      ("m,matrix", "Path to a Matrix Market (.mtx) file; omit to generate an ER graph", cxxopts::value<std::string>())
      ("v,vertices", "Vertex count for the generated ER graph", cxxopts::value<std::int64_t>()->default_value("50000"))
      ("d,degree", "Expected degree for the generated ER graph", cxxopts::value<double>()->default_value("16.0"))
      ("p,precision", "Numeric type: 'float', 'double', or 'both'", cxxopts::value<std::string>()->default_value("double"))
      ("index", "Index type: 'int32', 'int64', or 'both'", cxxopts::value<std::string>()->default_value(default_index))
      ("offset", "Row-offset type: 'int32', 'int64', or 'both'", cxxopts::value<std::string>()->default_value(default_offset))
      ("t,threads", "Comma-separated thread counts, e.g. '1,2,4,8'; empty means powers of two up to the maximum", cxxopts::value<std::string>()->default_value(""))
      ("s,seed", "Random seed for the graph generator and input vector", cxxopts::value<int>()->default_value("42"))
      ("i,iterations", "Maximum timed iterations per configuration", cxxopts::value<int>()->default_value("1000"))
      ("max-time", "Wall-clock budget per configuration in seconds", cxxopts::value<double>()->default_value("10.0"))
      ("o,output", "Write the raw per-iteration log to this JSON file", cxxopts::value<std::string>()->default_value(""))
      ("h,help", "Print help");
  // clang-format on
}

//! Parse "1,2,4"; an empty argument yields powers of two up to OMP_GET_MAX_THREADS().
[[nodiscard]] inline std::vector<int> ParseThreadCounts(const std::string& argument)
{
  std::vector<int> threads;
  std::stringstream stream(argument);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.find_first_not_of(" \t") == std::string::npos) continue;
    const int count = std::stoi(token);
    if (count <= 0) {
      throw std::invalid_argument("thread counts must be positive");
    }
    threads.push_back(count);
  }
  if (threads.empty()) {
    const int maximum = OMP_GET_MAX_THREADS();
    for (int count = 1; count < maximum; count *= 2) {
      threads.push_back(count);
    }
    threads.push_back(maximum);
  }
  return threads;
}

//! "int32", "int64", "float" or "double"; the name a report row carries.
template <class T>
[[nodiscard]] constexpr const char* TypeName()
{
  if constexpr (std::is_same_v<T, std::int32_t>) {
    return "int32";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return "int64";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, double>) {
    return "double";
  } else {
    return "unknown";
  }
}

//! "FP32" or "FP64", the shorter numeric label used in the timing table.
template <class NT>
[[nodiscard]] constexpr const char* PrecisionName()
{
  return std::is_same_v<NT, float> ? "FP32" : "FP64";
}

//! Compact "i32/o64" tag naming the index and offset types of a variant.
template <class IT, class OT>
[[nodiscard]] inline std::string IndexOffsetName()
{
  // "int32" -> "32": only the width varies, so drop the shared "int" prefix.
  const std::string index = TypeName<IT>() + 3;
  const std::string offset = TypeName<OT>() + 3;
  return "i" + index + "/o" + offset;
}

//! True when --precision selects NT. Throws on an unrecognized value.
template <class NT>
[[nodiscard]] inline bool PrecisionSelected(const std::string& precision)
{
  if (precision != "float" && precision != "double" && precision != "both") {
    throw std::invalid_argument("precision must be 'float', 'double', or 'both'");
  }
  if (precision == "both") return true;
  return precision == TypeName<NT>();
}

//! True when an --index or --offset argument selects T. Throws on a bad value.
template <class T>
[[nodiscard]] inline bool IntegerTypeSelected(const std::string& option)
{
  if (option != "int32" && option != "int64" && option != "both") {
    throw std::invalid_argument("index and offset types must be 'int32', 'int64', or 'both'");
  }
  if (option == "both") return true;
  return option == TypeName<T>();
}

/**
 * @brief True when every one of --index, --precision and --offset selects this variant.
 *
 * The Boost.PP product instantiates all eight combinations; this is what decides
 * which of them actually run.
 */
template <class IT, class NT, class OT>
[[nodiscard]] inline bool VariantSelected(const cxxopts::ParseResult& result)
{
  return IntegerTypeSelected<IT>(result["index"].as<std::string>()) &&
         PrecisionSelected<NT>(result["precision"].as<std::string>()) &&
         IntegerTypeSelected<OT>(result["offset"].as<std::string>());
}

//! Load the matrix named by --matrix, or generate an ER graph from --vertices/--degree.
template <class IT, class NT, class OT>
[[nodiscard]] MatrixInput<IT, NT, OT> LoadMatrix(const cxxopts::ParseResult& result)
{
  using Csr = spcraft::CsrMatrix<IT, NT, OT>;
  if (result.count("matrix") > 0) {
    const auto path = result["matrix"].as<std::string>();
    auto coo = spcraft::CooMatrix<IT, NT, OT>::FromMatrixMarket(path);
    return {std::make_shared<Csr>(coo.ToCsr()), path.substr(path.find_last_of("/\\") + 1)};
  }

  const auto vertices = static_cast<IT>(result["vertices"].as<std::int64_t>());
  const auto degree = result["degree"].as<double>();
  const auto seed = static_cast<std::uint64_t>(result["seed"].as<int>());
  std::ostringstream name;
  name << "ER_V" << vertices << "_D" << static_cast<long>(degree);
  return {std::make_shared<Csr>(spcraft::GenERGraph<NT, IT, OT>(vertices, degree, seed)),
          name.str()};
}

/**
 * @brief Bytes a single SpMV must move, assuming no reuse of the gathered x.
 *
 * This is an upper bound: it charges one x element per nonzero, so measured
 * bandwidth reads low on matrices whose columns cache well.
 */
template <class IT, class NT, class OT>
[[nodiscard]] inline double SpmvBytes(const spcraft::CsrMatrix<IT, NT, OT>& A)
{
  return static_cast<double>(A.nnz) * (2 * sizeof(NT) + sizeof(IT)) +
         static_cast<double>(A.m + 1) * sizeof(OT) + static_cast<double>(A.m) * sizeof(NT);
}

//! Relative tolerance for comparing a kernel result against the Eigen reference.
template <class NT>
[[nodiscard]] constexpr double Tolerance()
{
  return std::is_same_v<NT, float> ? 1.0e-4 : 1.0e-11;
}

/**
 * @brief Copy a SpCraft CSR matrix into an Eigen sparse matrix.
 *
 * Eigen is the independent reference these benchmarks check themselves against,
 * so this deliberately rebuilds from triplets rather than mapping the CSR
 * buffers: a shared view would let an indexing mistake agree with itself.
 * Duplicate entries are summed, matching CSR assembly.
 *
 * Eigen's default StorageIndex is `int` whatever IT is, so a 64-bit-indexed
 * matrix is checked here only up to that range; a larger one is rejected rather
 * than silently truncated.
 */
template <class IT, class NT, class OT>
[[nodiscard]] inline Eigen::SparseMatrix<NT, Eigen::RowMajor> ToEigen(
    const spcraft::CsrMatrix<IT, NT, OT>& A)
{
  constexpr auto kEigenLimit = std::numeric_limits<int>::max();
  if (static_cast<std::int64_t>(A.m) > kEigenLimit ||
      static_cast<std::int64_t>(A.n) > kEigenLimit) {
    throw std::runtime_error("matrix exceeds the Eigen reference's 32-bit index range");
  }

  std::vector<Eigen::Triplet<NT>> triplets;
  triplets.reserve(static_cast<std::size_t>(A.nnz));
  for (IT row = 0; row < A.m; ++row) {
    for (OT position = A.row_ptr[row]; position < A.row_ptr[row + 1]; ++position) {
      triplets.emplace_back(static_cast<int>(row), static_cast<int>(A.col_id[position]),
                            A.val[position]);
    }
  }
  Eigen::SparseMatrix<NT, Eigen::RowMajor> result(static_cast<int>(A.m), static_cast<int>(A.n));
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}

// //! View a SpCraft dense vector as an Eigen column vector without copying.
// template <class IT, class NT>
// [[nodiscard]] inline Eigen::Map<const Eigen::Matrix<NT, Eigen::Dynamic, 1>> ToEigen(
//     const spcraft::DenseVector<IT, NT>& x)
// {
//   return Eigen::Map<const Eigen::Matrix<NT, Eigen::Dynamic, 1>>(x.val, x.n);
// }

/**
 * @brief Largest absolute deviation of `y` from Eigen's product, throwing if too large.
 *
 * Shared by every driver so that "verified against Eigen" means the same test
 * regardless of backend or type combination.
 */
template <class IT, class NT, class OT>
[[nodiscard]] inline double VerifyAgainstEigen(const spcraft::CsrMatrix<IT, NT, OT>& A,
                                               const spcraft::DenseVector<IT, NT>& x, const NT* y,
                                               const char* backend)
{
  const Eigen::Matrix<NT, Eigen::Dynamic, 1> expected = ToEigen(A) * ToEigen(x);

  double largest = 0.0;
  double error = 0.0;
  for (IT row = 0; row < A.m; ++row) {
    largest = std::max(largest, std::abs(static_cast<double>(expected[row])));
    error = std::max(error, std::abs(static_cast<double>(expected[row] - y[row])));
  }
  const double tolerance = Tolerance<NT>() * std::max(1.0, largest);
  if (error > tolerance) {
    throw std::runtime_error(
        std::string(backend) + " SpMV disagrees with the Eigen reference: max error " +
        std::to_string(error) + " exceeds tolerance " + std::to_string(tolerance));
  }
  return error;
}

/**
 * @brief Run the kernel a few times, returning the median seconds per call.
 *
 * This warms the data caches and, more importantly, forces the OpenMP thread
 * team to be created at the current team size. The first parallel region after
 * a team resize pays that cost, and it would otherwise land inside the timed
 * loop. The returned estimate is what sizes the timed run.
 */
template <class Body>
[[nodiscard]] inline double WarmUp(Body&& body, int runs = 3)
{
  const std::vector<double> samples = spcraft::TimeIterations(body, runs);
  return spcraft::TimingStats::From(samples).median;
}

/**
 * @brief Timed iterations: at most max_iterations, and fitting the time budget.
 *
 * A configuration too slow to complete a single call inside the budget cannot be
 * measured meaningfully, so it is an error rather than a very long run.
 */
[[nodiscard]] inline int PlanIterations(double seconds_per_call, int max_iterations, double budget,
                                        const std::string& label)
{
  if (seconds_per_call > budget) {
    throw std::runtime_error(label + ": a single iteration takes " +
                             std::to_string(seconds_per_call) + "s, which exceeds the " +
                             std::to_string(budget) + "s budget; raise --max-time");
  }
  const auto affordable = static_cast<long long>(budget / seconds_per_call);
  return static_cast<int>(std::clamp<long long>(affordable, 1, max_iterations));
}

/**
 * @brief Fill in the fields of a run that come from the matrix and the types.
 *
 * Every driver records the same identifying columns for a variant, so they are
 * set in one place rather than copied into each driver's row assembly.
 */
template <class IT, class NT, class OT>
inline void DescribeRun(spcraft::BenchmarkRun& run, const spcraft::CsrMatrix<IT, NT, OT>& A,
                        const std::string& dataset, const char* backend)
{
  run.kernel = "SpMV";
  run.backend = backend;
  run.dataset = dataset;
  run.precision = PrecisionName<NT>();
  run.index_type = TypeName<IT>();
  run.offset_type = TypeName<OT>();
  run.rows = static_cast<std::int64_t>(A.m);
  run.columns = static_cast<std::int64_t>(A.n);
  run.nonzeros = static_cast<std::int64_t>(A.nnz);
  run.flops_per_iteration = 2.0 * static_cast<double>(A.nnz);
  run.bytes_per_iteration = SpmvBytes(A);
}

}  // namespace spcraft::benchmarks
