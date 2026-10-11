#include <fmt/format.h>
#include <mkl.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "BenchmarkCommon.h"
#include "common/BenchmarkOptions.h"
#include "mkl/MKLOptions.h"
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

/**
 * @brief Load, verify and time one <IT, NT, OT> combination against oneMKL.
 *
 * oneMKL's inspector partitions work using the thread count active at
 * optimization time, so a handle is rebuilt for every thread count rather than
 * reused across the sweep.
 */
template <class IT, class NT, class OT>
void RunVariant(const BenchmarkOptions& benchmark, const MKLOptions& options,
                spcraft::BenchmarkReport& report)
{
  if constexpr (kMklRepresentable<IT, OT>) {
    if (!VariantSelected<IT, NT, OT>(options.types)) return;

    // Matrix conversion can enter OpenMP. Keep it from creating a larger active
    // team whose idle CPU time would leak into the first benchmark setting.
    OMP_SET_NUM_THREADS(options.thread_counts.front());
    mkl_set_dynamic(0);
    mkl_set_num_threads(options.thread_counts.front());
    auto input = LoadMatrix<IT, NT, OT>(options.matrix);
    const spcraft::CsrMatrix<IT, NT, OT>& A = *input.matrix;
    spcraft::DenseVector<IT, NT> x(A.n);
    spcraft::DenseVector<IT, NT> y(A.m);
    x.Random(options.matrix.seed);

    {
      const spcraft::MklCsrSpmv<IT, NT, OT> handle(A);
      handle.Multiply(x.val, y.val);
    }
    const double error = VerifyAgainstEigen(A, x, y.val, "oneMKL");
    for (int threads : options.thread_counts) {
      const std::string label =
          fmt::format("SpMV/oneMKL/{}/{}/{}/threads:{}", input.name, IndexOffsetName<IT, OT>(),
                      PrecisionName<NT>(), threads);

      // Build the handle after fixing the thread count: mkl_sparse_optimize
      // partitions for the team it sees, and a later thread change does not
      // repartition it.
      OMP_SET_NUM_THREADS(threads);
      mkl_set_dynamic(0);
      mkl_set_num_threads(threads);
      const spcraft::MklCsrSpmv<IT, NT, OT> handle(A, benchmark.max_iterations);

      const auto once = [&] { handle.Multiply(x.val, y.val); };
      const double per_iteration = WarmUp(once);
      const int iterations = PlanIterations(per_iteration, benchmark.max_iterations,
                                            benchmark.max_time_seconds, label);

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
    const auto parsed = ParseMKLSpMVOptions(argc, argv);
    if (!parsed) return 0;
    const BenchmarkOptions& benchmark = parsed->benchmark;
    const MKLOptions& options = parsed->input;

    OMP_SET_DYNAMIC(0);

    std::string thread_list;
    for (std::size_t i = 0; i < options.thread_counts.size(); ++i) {
      thread_list += (i == 0 ? "" : ", ") + std::to_string(options.thread_counts[i]);
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
        {"index",
         fmt::format("{} (oneMKL supports {} only)", options.types.index, TypeName<MKL_INT>())},
        {"numeric", options.types.precision},
        {"offset",
         fmt::format("{} (oneMKL supports {} only)", options.types.offset, TypeName<MKL_INT>())},
        {"threads", "[" + thread_list + "]"},
        {"max iterations", std::to_string(benchmark.max_iterations)},
        {"time budget", fmt::format("{:.2f}s", benchmark.max_time_seconds)}};
    spcraft::BenchmarkReport report(std::move(report_info));

    RunVariant<MKL_INT, float, MKL_INT>(benchmark, options, report);
    RunVariant<MKL_INT, double, MKL_INT>(benchmark, options, report);

    if (report.runs().empty()) {
      throw std::runtime_error(
          fmt::format("no type combination selected; this oneMKL build represents CSR with "
                      "MKL_INT ({}) for both index and offset, so --index and --offset must "
                      "select it",
                      TypeName<MKL_INT>()));
    }

    const spcraft::TextReportRenderer text_renderer;
    spcraft::PrintReportToStderr(text_renderer.RenderSummary(report));

    const auto text_output = benchmark.text_report_path;
    if (!text_output.empty()) {
      spcraft::WriteReportFile(text_output, text_renderer.Render(report));
    }

    const auto output = benchmark.output_path;
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
