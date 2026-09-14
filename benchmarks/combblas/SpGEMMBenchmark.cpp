#include <CombBLAS/CombBLAS.h>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "SpCraft.h"
#include "SpGEMMBenchmarkCommon.h"

#ifndef THREADED
#error "The CombBLAS benchmark requires OpenMP"
#endif

namespace
{
using namespace spcraft::benchmarks;

template <class IT, class NT>
combblas::SpDCCols<IT, NT> ToCombBLAS(const spcraft::DcscMatrix<IT, NT>& matrix)
{
  combblas::SpTuples<IT, NT> tuples(matrix.nnz, matrix.m, matrix.n);
  for (IT col = 0; col < matrix.nzc; ++col) {
    for (IT p = matrix.col_ptr[col]; p < matrix.col_ptr[col + 1]; ++p) {
      tuples.rowindex(p) = matrix.row_id[p];
      tuples.colindex(p) = matrix.col_id[col];
      tuples.numvalue(p) = matrix.val[p];
    }
  }
  return combblas::SpDCCols<IT, NT>(tuples, false);
}

template <class IT, class NT>
void Run(const cxxopts::ParseResult& options, spcraft::BenchmarkReport& report)
{
  if (!VariantSelected<IT, NT, IT>(options)) return;
  using Ring = spcraft::PlusTimesRing<NT>;
  using CombRing = combblas::PlusTimesSRing<NT, NT>;
  using Tuples = combblas::SpTuples<IT, NT>;
  using DCSC = spcraft::DcscMatrix<IT, NT>;
  MatrixInput<IT, NT, IT> input;
  if (!options.count("matrix") && options["generator"].as<std::string>() == "rmat") {
    const int scale = options["scale"].as<int>();
    const auto edges = options["edge-factor"].as<std::size_t>();
    input = {std::make_shared<spcraft::CsrMatrix<IT, NT>>(spcraft::GenRMAT<NT, IT>(
                 static_cast<IT>(scale), edges, options["seed"].as<int>())),
             fmt::format("RMAT_S{}_E{}", scale, edges)};
  } else {
    if (!options.count("matrix") &&
        options["vertices"].as<std::int64_t>() > std::numeric_limits<IT>::max()) {
      throw std::overflow_error("vertex count exceeds index type");
    }
    input = LoadMatrix<IT, NT, IT>(options);
  }
  Eigen::SparseMatrix<NT> ea = ToEigen(*input.matrix);
  Eigen::SparseMatrix<NT> eb = ea;
  std::string dataset = input.name + " squared";
  if (options.count("right-matrix")) {
    const auto path = options["right-matrix"].as<std::string>();
    eb = ToEigen(spcraft::CooMatrix<IT, NT>::FromMatrixMarket(path).ToCsr());
    dataset = input.name + " x " + path.substr(path.find_last_of("/\\") + 1);
  }
  const int stride = options["column-stride"].as<int>();
  if (stride < 1) throw std::invalid_argument("column-stride must be positive");
  if (stride > 1) {
    // Embed the same graph in a larger shape to exercise genuinely hypersparse
    // columns without changing its scalar products or output nonzero count.
    auto embed = [stride](const Eigen::SparseMatrix<NT>& source) {
      const auto limit = std::min<std::int64_t>(std::numeric_limits<IT>::max(),
                                                std::numeric_limits<int>::max());
      if (source.rows() > limit / stride || source.cols() > limit / stride) {
        throw std::overflow_error("embedded shape exceeds index type");
      }
      Eigen::SparseMatrix<NT> result(source.rows() * stride, source.cols() * stride);
      std::vector<Eigen::Triplet<NT>> entries;
      entries.reserve(source.nonZeros());
      for (int col = 0; col < source.outerSize(); ++col) {
        for (typename Eigen::SparseMatrix<NT>::InnerIterator entry(source, col); entry;
             ++entry) {
          entries.emplace_back(entry.row() * stride, entry.col() * stride, entry.value());
        }
      }
      result.setFromTriplets(entries.begin(), entries.end());
      return result;
    };
    ea = embed(ea);
    eb = embed(eb);
    dataset += fmt::format(" stride{}", stride);
  }
  const auto a = FromEigenColumns<IT>(ea), b = FromEigenColumns<IT>(eb);
  if (a.n != b.m) throw std::invalid_argument("incompatible input dimensions");
  const Eigen::SparseMatrix<NT, Eigen::RowMajor> ra = ea, rb = eb;
  const auto csr_a = FromEigen<IT>(ra), csr_b = FromEigen<IT>(rb);
  const auto work = ProductWork(csr_a, csr_b);
  // Upstream also hashes signed keys and doubles signed symbolic capacities.
  // Keep the reference within those limits; our kernel's separate tests cover larger keys.
  constexpr auto kHashScale = 107;
  const auto upstream_limit = std::numeric_limits<IT>::max();
  if (a.m > upstream_limit / kHashScale || a.n == upstream_limit ||
      work > static_cast<std::uintmax_t>(upstream_limit / 2)) {
    throw std::overflow_error("input exceeds safe CombBLAS hash limits for this index type");
  }
  const auto da = DCSC::FromCsc(a), db = DCSC::FromCsc(b);
  const auto ca = ToCombBLAS(da), cb = ToCombBLAS(db);
  const Eigen::SparseMatrix<NT> expected = ea * eb;
  const auto expected_csc = FromEigenColumns<IT>(expected);
  const auto expected_dcsc = DCSC::FromCsc(expected_csc);
  auto native_combblas = [&] {
    return std::unique_ptr<Tuples>(
        combblas::LocalSpGEMMHash<CombRing, NT>(ca, cb, false, false, true));
  };
  report.AddInfo(dataset + "/" + PrecisionName<NT>() + "/output nnz",
                 std::to_string(expected.nonZeros()));
  fmt::print("{}: A {}x{}, {} nnz/{} nonempty columns; B {}x{}, {} nnz/{} nonempty columns\n",
             dataset, a.m, a.n, a.nnz, da.nzc, b.m, b.n, b.nnz, db.nzc);

  for (const int threads : ParseThreadCounts(options["threads"].as<std::string>())) {
    OMP_SET_NUM_THREADS(threads);
    std::array<double, 2> errors{};
    {
      const auto u = spcraft::OmpHashSpGEMM<Ring>(da, db);
      const auto t = native_combblas();
      if (u.nnz != expected.nonZeros() || t->getnnz() != expected.nonZeros())
        throw std::runtime_error("output nnz disagrees with Eigen");
      for (IT col = 0; col < expected_dcsc.nzc; ++col) {
        for (IT p = expected_dcsc.col_ptr[col]; p < expected_dcsc.col_ptr[col + 1]; ++p) {
          if (std::get<0>(u.entries[p]) != expected_dcsc.row_id[p] ||
              std::get<1>(u.entries[p]) != expected_dcsc.col_id[col] ||
              t->rowindex(p) != expected_dcsc.row_id[p] ||
              t->colindex(p) != expected_dcsc.col_id[col])
            throw std::runtime_error("output coordinate mismatch");
        }
      }
      errors[0] = Verify(
          expected, [&](Eigen::Index p) { return std::get<2>(u.entries[p]); },
          "OmpHashSpGEMM");
      errors[1] = Verify(expected, [&](Eigen::Index p) { return t->numvalue(p); }, "CombBLAS");
    }
    auto native = [&] {
      const auto c = spcraft::OmpHashSpGEMM<Ring>(da, db);
      if (c.nnz != expected.nonZeros()) std::abort();
    };
    auto reference = [&] {
      const auto c = native_combblas();
      if (c->getnnz() != expected.nonZeros()) std::abort();
    };
    auto sample = [&](std::size_t backend) {
      return backend == 0 ? spcraft::TimeIterations(native, 1).front()
                          : spcraft::TimeIterations(reference, 1).front();
    };
    constexpr std::size_t kBackends = 2;
    constexpr int kWarmups = 3;
    for (int i = 0; i < kWarmups; ++i)
      for (std::size_t backend = 0; backend < kBackends; ++backend) sample(backend);
    std::array<spcraft::BenchmarkRun, kBackends> runs;
    const std::array<const char*, kBackends> names = {"SPCraft-Hash-Tuples",
                                                      "CombBLAS-Hash-Tuples"};
    for (std::size_t i = 0; i < kBackends; ++i) {
      DescribeRun(runs[i], csr_a, dataset, names[i]);
      runs[i].kernel = "SpGEMM";
      runs[i].threads = threads;
      runs[i].columns = b.n;
      runs[i].flops_per_iteration = 2.0 * static_cast<double>(work);
      runs[i].bytes_per_iteration = 0;
      runs[i].verification_error = errors[i];
    }
    const auto start = std::chrono::steady_clock::now();
    // Alternate AB/BA so each kernel occupies both positions equally.
    for (int iteration = 0; iteration < options["iterations"].as<int>(); ++iteration) {
      for (std::size_t position = 0; position < kBackends; ++position) {
        const auto backend = (position + iteration % 2) % kBackends;
        runs[backend].seconds.push_back(sample(backend));
      }
      const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
      if (elapsed.count() >= options["max-time"].as<double>()) break;
    }
    for (std::size_t backend = 0; backend < kBackends; ++backend) {
      auto& run = runs[backend];
      fmt::print("{} / {} / {} threads: {:.3f} ms\n", dataset, run.backend, threads,
                 run.Stats().median * 1000);
      report.Add(std::move(run));
    }
    std::fflush(stdout);
  }
}
}  // namespace

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  try {
    int processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    if (processes != 1 || provided < MPI_THREAD_FUNNELED)
      throw std::runtime_error("requires one MPI rank with FUNNELED support");
    cxxopts::Options options("spgemm_combblas_benchmark",
                             "DCSC-to-COO OmpHashSpGEMM / CombBLAS comparison");
    AddCommonOptions(options, "int32", "int32");
    // clang-format off
    options.add_options()
      ("right-matrix", "Right operand; default A*A", cxxopts::value<std::string>())
      ("generator", "er or rmat", cxxopts::value<std::string>()->default_value("er"))
      ("column-stride", "Embed row/column IDs at this stride", cxxopts::value<int>()->default_value("1"))
      ("scale", "R-MAT scale", cxxopts::value<int>()->default_value("12"))
      ("edge-factor", "R-MAT edge factor", cxxopts::value<std::size_t>()->default_value("8"));
    // clang-format on
    const auto result = options.parse(argc, argv);
    if (result.count("help"))
      fmt::print("{}\n", options.help());
    else {
      if (result["iterations"].as<int>() < 1 || result["max-time"].as<double>() <= 0)
        throw std::invalid_argument("iterations and max-time must be positive");
      if (result["generator"].as<std::string>() != "er" &&
          result["generator"].as<std::string>() != "rmat")
        throw std::invalid_argument("generator must be er or rmat");
      OMP_SET_DYNAMIC(0);
      spcraft::BenchmarkReport report("OmpHashSpGEMM / CombBLAS native tuple comparison");
      report.AddInfo("SpCraft tuple kernel", "OmpHashSpGEMM (source-mapped baseline)");
      report.AddInfo("CombBLAS revision", SPCRAFT_COMBBLAS_REVISION);
      report.AddInfo("compiler", __VERSION__);
      report.AddInfo("build type", SPCRAFT_BENCHMARK_BUILD_TYPE);
      report.AddInfo("orientation",
                     "All implementations compute A*B directly by column expansion");
      report.AddInfo("timing",
                     "symbolic + numeric + sorting + allocation + destruction; "
                     "input conversions excluded; native tuple output on both sides");
      report.AddInfo("tuple output",
                     "Both native paths use std::tuple<IT,IT,NT>, sorted by column then row; "
                     "initialized array allocation and destruction included");
      report.AddInfo("measurement order", "native tuples only, paired AB/BA");
      report.AddInfo("reference", "Eigen full pattern and values at every thread count");
      report.PrintHeader();
      Run<std::int32_t, float>(result, report);
      Run<std::int32_t, double>(result, report);
      Run<std::int64_t, float>(result, report);
      Run<std::int64_t, double>(result, report);
      if (report.runs().empty())
        throw std::invalid_argument("use matching index/offset widths");
      report.PrintFooter();
      const auto path = result["output"].as<std::string>();
      if (!path.empty()) report.WriteJson(path);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    status = 1;
  }
  MPI_Finalize();
  return status;
}
