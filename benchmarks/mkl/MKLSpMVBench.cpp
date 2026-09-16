#include <cxxopts.hpp>
#include <fmt/format.h>
#include <mkl.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
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
 * 32-bit combination exists. The entry point instantiates only that supported
 * width, while this check keeps the backend constraint next to the implementation.
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

    // Matrix conversion can enter OpenMP. Keep it from creating a larger active
    // team whose idle CPU time would leak into the first benchmark setting.
    OMP_SET_NUM_THREADS(settings.thread_counts.front());
    mkl_set_dynamic(0);
    mkl_set_num_threads(settings.thread_counts.front());
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
      run.info.thread_count = threads;
      run.info.parameters.emplace_back("rank count", "1");
      run.verification = spcraft::VerificationResult{"Eigen sparse product", error, true};

      spcraft::HostTimingSamples timings = spcraft::TimeHostIterations(once, iterations);
      spcraft::GaugeSeries effective_cores =
          spcraft::metric::EffectiveCpuCores(timings.wall_seconds, timings.process_cpu_seconds);
      spcraft::GaugeSeries cpu_occupancy =
          spcraft::metric::CpuOccupancy(timings.wall_seconds, timings.process_cpu_seconds, threads);
      run.profiles.push_back(spcraft::ResourceProfile{
          spcraft::RunScope{}, {}, {spcraft::metric::WallTime(timings.wall_seconds)}});
      run.profiles.push_back(AlgorithmStorageProfile(A));
      run.profiles.push_back(spcraft::ResourceProfile{
          spcraft::CpuScope{0},
          {{"runtime", "oneMKL"}, {"requested_threads", std::to_string(threads)}},
          {spcraft::metric::ProcessCpuTime(std::move(timings.process_cpu_seconds)),
           std::move(effective_cores), std::move(cpu_occupancy)}});
      report.AddRun(std::move(run));
    }
  }
}

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

    spcraft::ReportInfo report_info;
    report_info.title = "oneMKL SpMV";
    report_info.metadata = {
        {"kernel", "mkl_sparse_?_mv (CSR, inspector-executor)"},
        {"reference", "Eigen sparse product, every row verified"},
        {"oneMKL", fmt::format("{}.{}.{} (GNU OpenMP, LP64)", version.MajorVersion,
                               version.MinorVersion, version.UpdateVersion)},
        {"index", fmt::format("{} (oneMKL supports {} only)", result["index"].as<std::string>(),
                              TypeName<MKL_INT>())},
        {"numeric", result["precision"].as<std::string>()},
        {"offset", fmt::format("{} (oneMKL supports {} only)", result["offset"].as<std::string>(),
                               TypeName<MKL_INT>())},
        {"threads", "[" + thread_list + "]"},
        {"max iterations", std::to_string(settings.max_iterations)},
        {"time budget", fmt::format("{:.2f}s", settings.budget)}};
    spcraft::BenchmarkReport report(std::move(report_info));

    RunVariant<MKL_INT, float, MKL_INT>(result, settings, report);
    RunVariant<MKL_INT, double, MKL_INT>(result, settings, report);

    if (report.runs().empty()) {
      throw std::runtime_error(
          fmt::format("no type combination selected; this oneMKL build represents CSR with "
                      "MKL_INT ({}) for both index and offset, so --index and --offset must "
                      "select it",
                      TypeName<MKL_INT>()));
    }

    const spcraft::TextReportRenderer text_renderer;
    spcraft::PrintReportToStderr(text_renderer.RenderSummary(report));

    const auto text_output = result["text-report"].as<std::string>();
    if (!text_output.empty()) {
      spcraft::WriteReportFile(text_output, text_renderer.Render(report));
    }

    const auto output = result["output"].as<std::string>();
    if (!output.empty()) {
      spcraft::WriteReportFile(output, spcraft::JsonReportRenderer{}.Render(report));
      fmt::print(stderr, "raw per-iteration log written to {}\n", output);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
  return 0;
}
