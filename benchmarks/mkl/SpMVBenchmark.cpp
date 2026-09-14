#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <mkl.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "BenchmarkCommon.h"
#include "SpCraft.h"

namespace
{

using namespace spcraft::benchmarks;

/**
 * @brief True when oneMKL can represent a CSR matrix with these index types.
 *
 * mkl_sparse_?_create_csr takes MKL_INT for both the row offsets and the column
 * indices, so under the LP64 interface this build links against, only the
 * 32-bit combination exists. The other six variants of the shared product are
 * skipped here rather than removed from the sequence, so that what oneMKL cannot
 * do stays visible next to what it can.
 */
template <class IT, class OT>
inline constexpr bool kMklRepresentable =
    std::is_same_v<IT, MKL_INT> && std::is_same_v<OT, MKL_INT>;

//! Everything a variant needs that does not depend on the type combination.
struct Settings {
  std::vector<int> thread_counts;
  int max_iterations = 0;
  double budget = 0.0;
  int seed = 0;
  bool header_printed = false;
};

/**
 * @brief Load, verify and time one <IT, NT, OT> combination against oneMKL.
 *
 * oneMKL's inspector partitions work using the thread count active at
 * optimization time, so a handle is rebuilt for every thread count rather than
 * reused across the sweep.
 */
template <class IT, class NT, class OT>
void RunVariant(const cxxopts::ParseResult& result, Settings& settings,
                spcraft::BenchmarkReport& report)
{
  if constexpr (kMklRepresentable<IT, OT>) {
    if (!VariantSelected<IT, NT, OT>(result)) return;

    auto input = LoadMatrix<IT, NT, OT>(result);
    const spcraft::CsrMatrix<IT, NT, OT>& A = *input.matrix;
    spcraft::DenseVector<IT, NT> x(A.n);
    spcraft::DenseVector<IT, NT> y(A.m);
    x.Random(settings.seed);

    {
      const spcraft::MklCsrSpmv<IT, NT, OT> handle(A);
      handle.Multiply(x.val, y.val);
    }
    const double error = VerifyAgainstEigen(A, x, y.val, "oneMKL");

    if (!settings.header_printed) {
      report.AddInfo(input.name, fmt::format("rows {}, cols {}, nnz {}", A.m, A.n, A.nnz));
      report.PrintHeader();
      settings.header_printed = true;
    }

    for (int threads : settings.thread_counts) {
      const std::string label =
          fmt::format("SpMV/oneMKL/{}/{}/{}/threads:{}", input.name, IndexOffsetName<IT, OT>(),
                      PrecisionName<NT>(), threads);

      // Build the handle after fixing the thread count: mkl_sparse_optimize
      // partitions for the team it sees, and a later thread change does not
      // repartition it.
      OMP_SET_NUM_THREADS(threads);
      mkl_set_dynamic(0);
      mkl_set_num_threads(threads);
      const spcraft::MklCsrSpmv<IT, NT, OT> handle(A, settings.max_iterations);

      const auto once = [&] { handle.Multiply(x.val, y.val); };
      const double per_iteration = WarmUp(once);
      const int iterations =
          PlanIterations(per_iteration, settings.max_iterations, settings.budget, label);

      spcraft::BenchmarkRun run;
      DescribeRun(run, A, input.name, "oneMKL");
      run.threads = threads;
      run.verification_error = error;
      run.seconds = spcraft::TimeIterations(once, iterations);
      report.Add(std::move(run));
    }
  }
}

#define SPCRAFT_BENCHMARK_RUN_VARIANT(r, product) \
  RunVariant<BOOST_PP_SEQ_ENUM(product)>(result, settings, report);

}  // namespace

int main(int argc, char** argv)
{
  try {
    cxxopts::Options options("spmv_mkl_benchmark",
                             "oneMKL SpMV benchmark over a thread-count sweep");
    // oneMKL's CSR is MKL_INT on both index and offset, so the shared 64-bit
    // offset default would select no variant at all here.
    AddCommonOptions(options, TypeName<MKL_INT>(), TypeName<MKL_INT>());
    const auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
      fmt::print("{}\n", options.help());
      return 0;
    }

    OMP_SET_DYNAMIC(0);

    Settings settings;
    settings.thread_counts = ParseThreadCounts(result["threads"].as<std::string>());
    settings.max_iterations = result["iterations"].as<int>();
    settings.budget = result["max-time"].as<double>();
    settings.seed = result["seed"].as<int>();

    std::string thread_list;
    for (std::size_t i = 0; i < settings.thread_counts.size(); ++i) {
      thread_list += (i == 0 ? "" : ", ") + std::to_string(settings.thread_counts[i]);
    }

    MKLVersion version;
    mkl_get_version(&version);

    spcraft::BenchmarkReport report("oneMKL SpMV");
    report.AddInfo("kernel", "mkl_sparse_?_mv (CSR, inspector-executor)");
    report.AddInfo("reference", "Eigen sparse product, every row verified");
    report.AddInfo("oneMKL", fmt::format("{}.{}.{} (GNU OpenMP, LP64)", version.MajorVersion,
                                         version.MinorVersion, version.UpdateVersion));
    report.AddInfo(
        "index", fmt::format("{} (oneMKL supports {} only)", result["index"].as<std::string>(),
                             TypeName<MKL_INT>()));
    report.AddInfo("numeric", result["precision"].as<std::string>());
    report.AddInfo("offset",
                   fmt::format("{} (oneMKL supports {} only)",
                               result["offset"].as<std::string>(), TypeName<MKL_INT>()));
    report.AddInfo("threads", "[" + thread_list + "]");
    report.AddInfo("max iterations", std::to_string(settings.max_iterations));
    report.AddInfo("time budget", fmt::format("{:.2f}s", settings.budget));

    BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_BENCHMARK_RUN_VARIANT,
                                  SPCRAFT_BENCHMARK_TYPE_SEQUENCES)

    if (!settings.header_printed) {
      throw std::runtime_error(
          fmt::format("no type combination selected; this oneMKL build represents CSR with "
                      "MKL_INT ({}) for both index and offset, so --index and --offset must "
                      "select it",
                      TypeName<MKL_INT>()));
    }
    report.PrintFooter();

    const auto output = result["output"].as<std::string>();
    if (!output.empty()) {
      report.WriteJson(output);
      fmt::print("raw per-iteration log written to {}\n", output);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
  return 0;
}

#undef SPCRAFT_BENCHMARK_RUN_VARIANT
