#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "SpCraft.h"

namespace
{

int failures = 0;

void Check(bool condition, const char* what)
{
  if (!condition) {
    std::cerr << "BenchmarkReport: " << what << "\n";
    ++failures;
  }
}

spcraft::BenchmarkRun SampleRun()
{
  spcraft::BenchmarkRun run;
  run.info.kernel = "SpMV";
  run.info.backend = "oneMKL";
  run.info.dataset = "sample";
  run.info.precision = "FP64";
  run.info.index_type = "int32";
  run.info.offset_type = "int32";
  run.info.thread_count = 4;
  run.info.parameters.emplace_back("rank count", "1");
  run.workload.rows = 10;
  run.workload.columns = 10;
  run.workload.input_nonzeros = 20;
  run.workload.flops_per_iteration = 40.0;
  run.workload.bytes_per_iteration = 640.0;
  run.verification = spcraft::VerificationResult{"closed form", 1.0e-12, true};
  run.profiles = {
      {spcraft::RunScope{},
       {},
       {spcraft::SecondsSeries{"wall_time", spcraft::MetricSummary::Median, {0.1, 0.2, 0.3}}}},
      {spcraft::RankScope{0},
       {{"host", "test-node"}},
       {spcraft::BytesSeries{"algorithm_storage", spcraft::MetricSummary::Maximum, {1024, 2048}}}},
      {spcraft::CpuScope{0},
       {{"runtime", "oneMKL"}},
       {spcraft::SecondsSeries{"process_cpu_time", spcraft::MetricSummary::Median, {0.3, 0.4}},
        spcraft::GaugeSeries{
            "effective_cpu_cores", "cores", spcraft::MetricSummary::Median, {3.0, 4.0}}}},
      {spcraft::GpuScope{0, 1},
       {{"name", "test GPU"}},
       {spcraft::BytesSeries{"device_memory", spcraft::MetricSummary::Maximum, {4096, 8192}}}},
      {spcraft::RankAggregateScope{spcraft::RankAggregation::MaxToMean},
       {},
       {spcraft::GaugeSeries{"rank_imbalance", "max/mean", spcraft::MetricSummary::Mean, {1.25}}}}};
  run.stages = {{"thread_profile",
                 {{"timing", "separate pass"}},
                 {{spcraft::RunScope{},
                   {},
                   {spcraft::GaugeSeries{
                       "wall_time_imbalance", "max/mean", spcraft::MetricSummary::Mean, {1.1}}}},
                  {spcraft::ThreadScope{0, 3},
                   {{"cpu_start", "2"}, {"cpu_end", "4"}},
                   {spcraft::SecondsSeries{
                       "thread_wall_time", spcraft::MetricSummary::Mean, {0.04, 0.05}}}}}}};
  return run;
}

template <class Function>
bool ThrowsInvalidArgument(Function&& function)
{
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main()
{
  {
    const spcraft::TimingStats stats = spcraft::TimingStats::From({0.3, 0.1, 0.2});
    Check(std::abs(stats.minimum - 0.1) < 1.0e-12, "statistics should record the minimum");
    Check(std::abs(stats.maximum - 0.3) < 1.0e-12, "statistics should record the maximum");
    Check(std::abs(stats.median - 0.2) < 1.0e-12, "statistics should compute the median");
    Check(std::abs(stats.mean - 0.2) < 1.0e-12, "statistics should compute the mean");
    Check(std::abs(stats.stddev - 0.1) < 1.0e-12,
          "statistics should use sample standard deviation");
  }

  spcraft::BenchmarkReport report(
      spcraft::ReportInfo{"Scoped benchmark report", {{"machine", "test-node"}}});
  report.AddRun(SampleRun());
  Check(report.runs().size() == 1, "AddRun should retain a validated run");

  const std::string text = spcraft::TextReportRenderer{}.Render(report);
  Check(text.find("oneMKL") != std::string::npos, "text should identify the backend");
  Check(text.find("rank:0/thread:3") != std::string::npos, "text should render a thread scope");
  Check(text.find("rank:0/cpu") != std::string::npos, "text should render a CPU scope");
  Check(text.find("rank:0/gpu:1") != std::string::npos, "text should render a GPU scope");
  Check(text.find("algorithm_storage") != std::string::npos,
        "text should render algorithm memory measurements");
  Check(text.find("stage thread_profile") != std::string::npos,
        "text should identify algorithm stages");
  Check(text.find("rank-aggregate:max-to-mean") != std::string::npos,
        "text should render rank aggregations");

  {
    spcraft::BenchmarkRun signed_gauge = SampleRun();
    signed_gauge.profiles[2].metrics.push_back(
        spcraft::GaugeSeries{"temperature_delta", "K", spcraft::MetricSummary::Mean, {-2.0, 1.0}});
    spcraft::BenchmarkReport gauge_report(spcraft::ReportInfo{"signed gauge", {}});
    gauge_report.AddRun(std::move(signed_gauge));
    Check(spcraft::TextReportRenderer{}.Render(gauge_report).find("temperature_delta") !=
              std::string::npos,
          "unconstrained gauges should render finite negative values");
  }

  const std::string summary = spcraft::TextReportRenderer{}.RenderSummary(report);
  Check(summary.find("PROFILE DETAILS") == std::string::npos,
        "summary rendering should omit detailed profiles");

  const std::string json = spcraft::JsonReportRenderer{}.Render(report);
  Check(json.find("\"schema_version\": 3") != std::string::npos,
        "JSON should carry its schema version");
  Check(json.find("\"median_seconds\": 0.2") != std::string::npos,
        "JSON should retain the established timing summary");
  Check(json.find("\"scope\": \"rank:0/thread:3\"") != std::string::npos,
        "JSON should retain scoped profile data");
  Check(json.find("\"name\": \"thread_profile\"") != std::string::npos,
        "JSON should retain algorithm stages");

  {
    const std::string path = "benchmark_report_test.txt";
    spcraft::WriteReportFile(path, text);
    std::ifstream input(path);
    const std::string saved{std::istreambuf_iterator<char>(input),
                            std::istreambuf_iterator<char>()};
    Check(saved == text, "the text file should exactly match the rendered report");
    std::remove(path.c_str());
  }

  {
    spcraft::BenchmarkRun invalid = SampleRun();
    auto& wall_time = std::get<spcraft::SecondsSeries>(invalid.profiles.front().metrics.front());
    wall_time.samples.clear();
    Check(ThrowsInvalidArgument([&] { spcraft::ValidateBenchmarkRun(invalid); }),
          "an empty metric series should be rejected");
  }

  {
    spcraft::BenchmarkRun invalid = SampleRun();
    invalid.profiles.push_back(invalid.profiles[1]);
    Check(ThrowsInvalidArgument([&] { spcraft::ValidateBenchmarkRun(invalid); }),
          "duplicate scopes should be rejected");
  }

  {
    spcraft::BenchmarkRun invalid = SampleRun();
    invalid.profiles[1].scope = spcraft::RankScope{-1};
    Check(ThrowsInvalidArgument([&] { spcraft::ValidateBenchmarkRun(invalid); }),
          "negative rank identifiers should be rejected");
  }

  {
    int calls = 0;
    const std::vector<double> samples = spcraft::TimeIterations([&] { ++calls; }, 3);
    Check(calls == 3 && samples.size() == 3,
          "TimeIterations should execute and record every requested iteration");
    Check(ThrowsInvalidArgument([&] { (void)spcraft::TimeIterations([] {}, 0); }),
          "TimeIterations should reject zero iterations");
  }

  {
    int calls = 0;
    const spcraft::HostTimingSamples samples = spcraft::TimeHostIterations([&] { ++calls; }, 3);
    Check(calls == 3 && samples.wall_seconds.size() == 3 && samples.process_cpu_seconds.size() == 3,
          "host timing should return paired wall and process CPU samples");
    const auto cores =
        spcraft::metric::EffectiveCpuCores(samples.wall_seconds, samples.process_cpu_seconds);
    Check(cores.samples.size() == 1,
          "effective CPU cores should aggregate paired samples without accounting spikes");
    Check(
        ThrowsInvalidArgument([&] { (void)spcraft::metric::EffectiveCpuCores({0.1}, {0.1, 0.2}); }),
        "derived paired metrics should reject mismatched sample counts");
  }

  if (failures != 0) {
    std::cerr << failures << " BenchmarkReport check(s) failed\n";
    return 1;
  }
  std::cout << "BenchmarkReport checks passed\n";
  return 0;
}
