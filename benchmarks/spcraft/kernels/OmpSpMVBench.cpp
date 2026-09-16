#include <cxxopts.hpp>
#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <vector>

#include "BenchmarkCommon.h"
#include "SpCraft.h"

using namespace spcraft::benchmarks;

template <class NT>
using Ring = spcraft::PlusTimesRing<NT>;

//! Everything a variant needs that does not depend on the type combination.
struct Settings {
  std::vector<int> thread_counts;
  int max_iterations = 0;
  double budget = 0.0;
  int seed = 0;
};

/**
 * @brief Load, verify and time one <IT, NT, OT> combination of the SpMV kernel.
 *
 * The matrix is rebuilt per variant rather than converted from a single load:
 * the index and offset widths are exactly what is under test, so sharing one
 * representation between variants would measure the wrong thing.
 *
 * Every selected variant publishes a complete run to the report after timing.
 */
template <class IT, class NT, class OT>
void RunVariant(const cxxopts::ParseResult& result, Settings& settings,
                spcraft::BenchmarkReport& report)
{
  if (!VariantSelected<IT, NT, OT>(result)) return;

  // Matrix conversion can enter OpenMP. Keep it from creating a larger active
  // team whose idle CPU time would leak into the first benchmark setting.
  OMP_SET_NUM_THREADS(settings.thread_counts.front());
  auto input = LoadMatrix<IT, NT, OT>(result);
  const spcraft::CsrMatrix<IT, NT, OT>& A = *input.matrix;
  spcraft::DenseVector<IT, NT> x(A.n);
  spcraft::DenseVector<IT, NT> y(A.m);
  x.Random(settings.seed);

  spcraft::OmpSpMV<Ring<NT>>(A, x, y);
  const double error = VerifyAgainstEigen(A, x, y.val, "SPCraft OpenMP");

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
    run.info.thread_count = threads;
    run.info.parameters.emplace_back("rank count", "1");
    run.verification = spcraft::VerificationResult{"Eigen sparse product", error, true};
    spcraft::HostTimingSamples timings =
        spcraft::TimeHostIterations([&] { spcraft::OmpSpMV<Ring<NT>>(A, x, y); }, iterations);
    spcraft::GaugeSeries effective_cores =
        spcraft::metric::EffectiveCpuCores(timings.wall_seconds, timings.process_cpu_seconds);
    spcraft::GaugeSeries cpu_occupancy =
        spcraft::metric::CpuOccupancy(timings.wall_seconds, timings.process_cpu_seconds, threads);
    run.profiles.push_back(spcraft::ResourceProfile{
        spcraft::RunScope{}, {}, {spcraft::metric::WallTime(timings.wall_seconds)}});
    run.profiles.push_back(AlgorithmStorageProfile(A));
    run.profiles.push_back(spcraft::ResourceProfile{
        spcraft::CpuScope{0},
        {{"runtime", "OpenMP"}, {"requested_threads", std::to_string(threads)}},
        {spcraft::metric::ProcessCpuTime(std::move(timings.process_cpu_seconds)),
         std::move(effective_cores), std::move(cpu_occupancy)}});
    report.AddRun(std::move(run));
  }
}

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

    spcraft::ReportInfo report_info;
    report_info.title = "SPCraft OpenMP SpMV";
    report_info.metadata = {{"kernel", "SpMV (PlusTimesRing, CSR)"},
                            {"reference", "Eigen sparse product, every row verified"},
                            {"index", result["index"].as<std::string>()},
                            {"numeric", result["precision"].as<std::string>()},
                            {"offset", result["offset"].as<std::string>()},
                            {"threads", "[" + thread_list + "]"},
                            {"max iterations", std::to_string(settings.max_iterations)},
                            {"time budget", fmt::format("{:.2f}s", settings.budget)}};
    spcraft::BenchmarkReport report(std::move(report_info));

    RunVariant<std::int32_t, float, std::int32_t>(result, settings, report);
    RunVariant<std::int32_t, float, std::int64_t>(result, settings, report);
    RunVariant<std::int32_t, double, std::int32_t>(result, settings, report);
    RunVariant<std::int32_t, double, std::int64_t>(result, settings, report);
    RunVariant<std::int64_t, float, std::int32_t>(result, settings, report);
    RunVariant<std::int64_t, float, std::int64_t>(result, settings, report);
    RunVariant<std::int64_t, double, std::int32_t>(result, settings, report);
    RunVariant<std::int64_t, double, std::int64_t>(result, settings, report);

    if (report.runs().empty()) {
      throw std::runtime_error(
          "no type combination selected; check --index, --precision and --offset");
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
