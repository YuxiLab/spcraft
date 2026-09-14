#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cxxopts.hpp>
#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <vector>

#include "BenchmarkCommon.h"
#include "SpCraft.h"

namespace
{

using namespace spcraft::benchmarks;

template <class NT>
using Ring = spcraft::PlusTimesRing<NT>;

//! Everything a variant needs that does not depend on the type combination.
struct Settings {
  std::vector<int> thread_counts;
  int max_iterations = 0;
  double budget = 0.0;
  int seed = 0;
  bool header_printed = false;
};

/**
 * @brief Load, verify and time one <IT, NT, OT> combination of the SpMV kernel.
 *
 * The matrix is rebuilt per variant rather than converted from a single load:
 * the index and offset widths are exactly what is under test, so sharing one
 * representation between variants would measure the wrong thing.
 *
 * The dataset banner is emitted by whichever variant runs first, because the
 * matrix shape is the same for all of them and the banner must precede the table.
 */
template <class IT, class NT, class OT>
void RunVariant(const cxxopts::ParseResult& result, Settings& settings,
                spcraft::BenchmarkReport& report)
{
  if (!VariantSelected<IT, NT, OT>(result)) return;

  auto input = LoadMatrix<IT, NT, OT>(result);
  const spcraft::CsrMatrix<IT, NT, OT>& A = *input.matrix;
  spcraft::DenseVector<IT, NT> x(A.n);
  spcraft::DenseVector<IT, NT> y(A.m);
  x.Random(settings.seed);

  spcraft::OmpSpMV<Ring<NT>>(A, x, y);
  const double error = VerifyAgainstEigen(A, x, y.val, "SPCraft OpenMP");

  if (!settings.header_printed) {
    report.AddInfo(input.name, fmt::format("rows {}, cols {}, nnz {}", A.m, A.n, A.nnz));
    report.PrintHeader();
    settings.header_printed = true;
  }

  for (int threads : settings.thread_counts) {
    const std::string label = fmt::format("SpMV/{}/{}/{}/threads:{}", input.name,
                                          IndexOffsetName<IT, OT>(), PrecisionName<NT>(), threads);

    // Warm up at this exact team size, then size the timed run from what it measured.
    OMP_SET_NUM_THREADS(threads);
    const double per_iteration = WarmUp([&] { spcraft::OmpSpMV<Ring<NT>>(A, x, y); });
    const int iterations =
        PlanIterations(per_iteration, settings.max_iterations, settings.budget, label);

    spcraft::BenchmarkRun run;
    DescribeRun(run, A, input.name, "SPCraft-OpenMP");
    run.threads = threads;
    run.verification_error = error;
    run.seconds = spcraft::TimeIterations([&] { spcraft::OmpSpMV<Ring<NT>>(A, x, y); }, iterations);
    report.Add(std::move(run));
  }
}

// The kernel is templated on all three parameters, so every combination of them
// is instantiated here and selected at run time by --index/--precision/--offset.
#define SPCRAFT_BENCHMARK_RUN_VARIANT(r, product) \
  RunVariant<BOOST_PP_SEQ_ENUM(product)>(result, settings, report);

}  // namespace

int main(int argc, char** argv)
{
  try {
    cxxopts::Options options("spmv_openmp_benchmark",
                             "SPCraft OpenMP SpMV benchmark over a thread-count sweep");
    AddCommonOptions(options);
    const auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
      fmt::print("{}\n", options.help());
      return 0;
    }

    // Fixed thread counts only; a dynamic team size would make the sweep meaningless.
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

    spcraft::BenchmarkReport report("SPCraft OpenMP SpMV");
    report.AddInfo("kernel", "SpMV (PlusTimesRing, CSR)");
    report.AddInfo("reference", "Eigen sparse product, every row verified");
    report.AddInfo("index", result["index"].as<std::string>());
    report.AddInfo("numeric", result["precision"].as<std::string>());
    report.AddInfo("offset", result["offset"].as<std::string>());
    report.AddInfo("threads", "[" + thread_list + "]");
    report.AddInfo("max iterations", std::to_string(settings.max_iterations));
    report.AddInfo("time budget", fmt::format("{:.2f}s", settings.budget));

    if (!settings.header_printed) {
      throw std::runtime_error(
          "no type combination selected; check --index, --precision and --offset");
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
