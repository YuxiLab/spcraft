#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "common/BenchmarkOptions.h"
#include "SpCraft.h"

namespace spcraft::benchmarks
{
using HostSamples = HostTimingSamples;

class GoogleBenchmarkSession
{
 public:
  GoogleBenchmarkSession()
  {
    // SpCraft's cxxopts parser owns the CLI; configure measurements through its options.
    int argc = 1;
    // Google Benchmark retains argv[0] for its reporter context.
    static char name[] = "spcraft_benchmark";
    char* argv[] = {name, nullptr};
    benchmark::Initialize(&argc, argv);
  }

  ~GoogleBenchmarkSession()
  {
    benchmark::ClearRegisteredBenchmarks();
    benchmark::Shutdown();
  }

  GoogleBenchmarkSession(const GoogleBenchmarkSession&) = delete;
  GoogleBenchmarkSession& operator=(const GoogleBenchmarkSession&) = delete;
};

class HostTimingReporter final : public benchmark::BenchmarkReporter
{
 public:
  HostSamples samples;
  std::string error;

  bool ReportContext(const Context&) override { return true; }

  void ReportRuns(const std::vector<Run>& runs) override
  {
    for (const auto& run : runs) {
      if (run.skipped != benchmark::internal::NotSkipped) {
        error = run.skip_message;
        continue;
      }
      if (run.run_type != Run::RT_Iteration) continue;
      const double iterations = static_cast<double>(run.iterations);
      samples.wall_seconds.push_back(run.real_accumulated_time / iterations);
      samples.process_cpu_seconds.push_back(run.cpu_accumulated_time / iterations);
    }
  }
};

template <class Body>
HostSamples CollectHostSamples(const std::string& name, Body& body, int repetitions)
{
  // One driver thread invokes the complete internally threaded kernel.
  const auto measured = [&](benchmark::State& state) {
    for (auto iteration : state) {
      body();
      benchmark::ClobberMemory();
    }
  };
  auto* measurement = benchmark::RegisterBenchmark(name.c_str(), measured);
  measurement->Threads(1)->Iterations(1)->Repetitions(repetitions);
  measurement->UseRealTime()->MeasureProcessCPUTime()->Unit(benchmark::kSecond);
  measurement->DisplayAggregatesOnly(false)->ReportAggregatesOnly(false);

  // Keep individual repetitions for SpCraft's raw wall/process CPU report schema.
  HostTimingReporter reporter;
  benchmark::RunSpecifiedBenchmarks(&reporter, ".*");
  benchmark::ClearRegisteredBenchmarks();
  if (!reporter.error.empty()) throw std::runtime_error(reporter.error);
  if (reporter.samples.wall_seconds.size() != static_cast<std::size_t>(repetitions)) {
    throw std::runtime_error("Google Benchmark did not produce the requested timing samples");
  }
  return std::move(reporter.samples);
}

template <class Body>
HostSamples TimeHostBenchmark(const std::string& name, Body body, const BenchmarkOptions& opts)
{
  // Discard three invocations to establish the selected kernel team and warm its caches.
  constexpr int kWarmupRuns = 3;
  const auto warmup = CollectHostSamples(name + "/warmup", body, kWarmupRuns);
  const double per_call = TimingStats::From(warmup.wall_seconds).median;
  if (per_call > opts.max_time_seconds) {
    throw std::runtime_error(name + ": a single invocation exceeds --max-time");
  }

  // Preserve the existing iteration cap and estimated wall-time budget per configuration.
  const double affordable = opts.max_time_seconds / per_call;
  const int repetitions =
      affordable >= opts.max_iterations ? opts.max_iterations : std::max(1, static_cast<int>(affordable));
  auto samples = CollectHostSamples(name, body, repetitions);
  return samples;
}
}  // namespace spcraft::benchmarks
