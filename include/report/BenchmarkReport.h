#pragma once

#include <fmt/format.h>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace spcraft
{

using ReportMetadata = std::vector<std::pair<std::string, std::string>>;

/** @brief Summary statistics over a nonempty set of samples. */
struct TimingStats {
  double minimum = 0.0;
  double maximum = 0.0;
  double median = 0.0;
  double mean = 0.0;
  double stddev = 0.0;

  [[nodiscard]] double CoefficientOfVariation() const
  {
    return mean > 0.0 ? 100.0 * stddev / mean : 0.0;
  }

  [[nodiscard]] static TimingStats From(const std::vector<double>& samples,
                                        bool allow_negative = false)
  {
    if (samples.empty()) {
      throw std::invalid_argument("TimingStats requires at least one sample");
    }
    for (double sample : samples) {
      if (!std::isfinite(sample) || (!allow_negative && sample < 0.0)) {
        throw std::invalid_argument("samples must be finite and satisfy their metric range");
      }
    }

    TimingStats stats;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    stats.minimum = sorted.front();
    stats.maximum = sorted.back();
    const std::size_t middle = sorted.size() / 2;
    stats.median =
        sorted.size() % 2 == 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
    stats.mean =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());

    if (sorted.size() > 1) {
      double sum_squares = 0.0;
      for (double value : sorted) {
        const double deviation = value - stats.mean;
        sum_squares += deviation * deviation;
      }
      stats.stddev = std::sqrt(sum_squares / static_cast<double>(sorted.size() - 1));
    }
    return stats;
  }
};

enum class MetricSummary { Minimum, Median, Mean, Maximum, Sum, Last };

struct SecondsSeries {
  std::string name;
  MetricSummary headline;
  std::vector<double> samples;
};

struct BytesSeries {
  std::string name;
  MetricSummary headline;
  std::vector<std::uint64_t> samples;
};

struct CountSeries {
  std::string name;
  MetricSummary headline;
  std::vector<std::uint64_t> samples;
};

//! Ratios are stored in [0, 1] and rendered as percentages.
struct RatioSeries {
  std::string name;
  MetricSummary headline;
  std::vector<double> samples;
};

//! A finite scalar with an explicit unit and no artificial range restriction.
struct GaugeSeries {
  std::string name;
  std::string unit;
  MetricSummary headline;
  std::vector<double> samples;
};

using MetricSeries =
    std::variant<SecondsSeries, BytesSeries, CountSeries, RatioSeries, GaugeSeries>;

namespace metric
{

inline constexpr std::string_view kWallTime = "wall_time";
inline constexpr std::string_view kProcessCpuTime = "process_cpu_time";
inline constexpr std::string_view kEffectiveCpuCores = "effective_cpu_cores";
inline constexpr std::string_view kCpuOccupancy = "cpu_occupancy";
inline constexpr std::string_view kAlgorithmStorage = "algorithm_storage";
inline constexpr std::string_view kThreadWallTime = "thread_wall_time";
inline constexpr std::string_view kThreadCpuTime = "thread_cpu_time";
inline constexpr std::string_view kAssignedRows = "assigned_rows";
inline constexpr std::string_view kAssignedNonzeros = "assigned_nonzeros";
inline constexpr std::string_view kWallTimeImbalance = "wall_time_imbalance";
inline constexpr std::string_view kCpuTimeImbalance = "cpu_time_imbalance";
inline constexpr std::string_view kNonzeroImbalance = "nonzero_imbalance";

[[nodiscard]] inline SecondsSeries WallTime(std::vector<double> samples)
{
  return {std::string(kWallTime), MetricSummary::Median, std::move(samples)};
}

[[nodiscard]] inline SecondsSeries ProcessCpuTime(std::vector<double> samples)
{
  // Linux may charge live worker threads to the process clock in batches.
  // The total (and therefore mean per iteration) remains correct, while a
  // per-call median can incorrectly look like main-thread CPU time only.
  return {std::string(kProcessCpuTime), MetricSummary::Mean, std::move(samples)};
}

[[nodiscard]] inline GaugeSeries EffectiveCpuCores(const std::vector<double>& wall_seconds,
                                                   const std::vector<double>& cpu_seconds)
{
  if (wall_seconds.size() != cpu_seconds.size()) {
    throw std::invalid_argument("wall and process CPU sample counts must match");
  }
  const double wall = std::accumulate(wall_seconds.begin(), wall_seconds.end(), 0.0);
  const double cpu = std::accumulate(cpu_seconds.begin(), cpu_seconds.end(), 0.0);
  return {std::string(kEffectiveCpuCores),
          "cores",
          MetricSummary::Mean,
          {wall > 0.0 ? cpu / wall : 0.0}};
}

[[nodiscard]] inline GaugeSeries CpuOccupancy(const std::vector<double>& wall_seconds,
                                              const std::vector<double>& cpu_seconds, int threads)
{
  if (threads < 1) throw std::invalid_argument("thread count must be positive");
  GaugeSeries cores = EffectiveCpuCores(wall_seconds, cpu_seconds);
  for (double& sample : cores.samples) sample /= static_cast<double>(threads);
  cores.name = kCpuOccupancy;
  cores.unit = "ratio";
  return cores;
}

}  // namespace metric

struct RunScope {
};

struct RankScope {
  int rank;
};

struct ThreadScope {
  int rank;
  int thread;
};

//! All CPU execution attributed to one rank; per-thread detail uses ThreadScope.
struct CpuScope {
  int rank;
};

struct GpuScope {
  int rank;
  int device;
};

enum class RankAggregation { Maximum, Sum, MaxToMean };

//! A cross-rank value produced by an MPI-aware collector outside the report model.
struct RankAggregateScope {
  RankAggregation aggregation;
};

using ProfileScope =
    std::variant<RunScope, RankScope, ThreadScope, CpuScope, GpuScope, RankAggregateScope>;

struct ResourceProfile {
  ProfileScope scope;
  ReportMetadata attributes;
  std::vector<MetricSeries> metrics;
};

/**
 * @brief Measurements for one named algorithm phase.
 *
 * Stage samples are intentionally not summed: phases may overlap or be nested.
 */
struct AlgorithmStage {
  std::string name;
  ReportMetadata attributes;
  std::vector<ResourceProfile> profiles;
};

struct ReportInfo {
  std::string title;
  ReportMetadata metadata;
};

struct RunInfo {
  std::string kernel;
  std::string backend;
  std::string dataset;
  std::string precision;
  std::string index_type;
  std::string offset_type;
  std::optional<int> thread_count;
  ReportMetadata parameters;
};

struct WorkloadInfo {
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t input_nonzeros = 0;
  std::optional<std::int64_t> output_nonzeros;
  double flops_per_iteration = 0.0;
  double bytes_per_iteration = 0.0;
};

struct VerificationResult {
  std::string reference;
  double maximum_error = 0.0;
  bool passed = false;
};

namespace detail
{

[[nodiscard]] inline const char* SummaryName(MetricSummary summary)
{
  switch (summary) {
    case MetricSummary::Minimum:
      return "min";
    case MetricSummary::Median:
      return "median";
    case MetricSummary::Mean:
      return "mean";
    case MetricSummary::Maximum:
      return "max";
    case MetricSummary::Sum:
      return "sum";
    case MetricSummary::Last:
      return "last";
  }
  throw std::logic_error("unknown metric summary");
}

template <class T>
[[nodiscard]] inline double Headline(const std::vector<T>& samples, MetricSummary summary,
                                     bool allow_negative = false)
{
  if (samples.empty()) {
    throw std::invalid_argument("metric series requires at least one sample");
  }
  std::vector<double> values;
  values.reserve(samples.size());
  for (T sample : samples) values.push_back(static_cast<double>(sample));
  const TimingStats stats = TimingStats::From(values, allow_negative);
  switch (summary) {
    case MetricSummary::Minimum:
      return stats.minimum;
    case MetricSummary::Median:
      return stats.median;
    case MetricSummary::Mean:
      return stats.mean;
    case MetricSummary::Maximum:
      return stats.maximum;
    case MetricSummary::Sum:
      return std::accumulate(values.begin(), values.end(), 0.0);
    case MetricSummary::Last:
      return values.back();
  }
  throw std::logic_error("unknown metric summary");
}

[[nodiscard]] inline std::string ScopeKey(const ProfileScope& scope)
{
  return std::visit(
      [](const auto& value) -> std::string {
        using Scope = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Scope, RunScope>) {
          return "run";
        } else if constexpr (std::is_same_v<Scope, RankScope>) {
          return fmt::format("rank:{}", value.rank);
        } else if constexpr (std::is_same_v<Scope, ThreadScope>) {
          return fmt::format("rank:{}/thread:{}", value.rank, value.thread);
        } else if constexpr (std::is_same_v<Scope, CpuScope>) {
          return fmt::format("rank:{}/cpu", value.rank);
        } else if constexpr (std::is_same_v<Scope, GpuScope>) {
          return fmt::format("rank:{}/gpu:{}", value.rank, value.device);
        } else {
          switch (value.aggregation) {
            case RankAggregation::Maximum:
              return "rank-aggregate:max";
            case RankAggregation::Sum:
              return "rank-aggregate:sum";
            case RankAggregation::MaxToMean:
              return "rank-aggregate:max-to-mean";
          }
          throw std::logic_error("unknown rank aggregation");
        }
      },
      scope);
}

inline void ValidateScope(const ProfileScope& scope)
{
  std::visit(
      [](const auto& value) {
        using Scope = std::decay_t<decltype(value)>;
        if constexpr (!std::is_same_v<Scope, RunScope> &&
                      !std::is_same_v<Scope, RankAggregateScope>) {
          if (value.rank < 0) throw std::invalid_argument("profile rank cannot be negative");
        }
        if constexpr (std::is_same_v<Scope, ThreadScope>) {
          if (value.thread < 0) throw std::invalid_argument("profile thread cannot be negative");
        }
        if constexpr (std::is_same_v<Scope, GpuScope>) {
          if (value.device < 0) throw std::invalid_argument("profile GPU cannot be negative");
        }
      },
      scope);
}

inline void ValidateMetric(const MetricSeries& metric)
{
  std::visit(
      [](const auto& series) {
        if (series.name.empty()) throw std::invalid_argument("metric name cannot be empty");
        if (series.samples.empty()) {
          throw std::invalid_argument("metric series requires at least one sample");
        }
        using Series = std::decay_t<decltype(series)>;
        if constexpr (std::is_same_v<Series, SecondsSeries>) {
          (void)TimingStats::From(series.samples);
        } else if constexpr (std::is_same_v<Series, RatioSeries>) {
          for (double sample : series.samples) {
            if (!std::isfinite(sample) || sample < 0.0 || sample > 1.0) {
              throw std::invalid_argument("ratio samples must be finite and in [0, 1]");
            }
          }
        } else if constexpr (std::is_same_v<Series, GaugeSeries>) {
          if (series.unit.empty()) throw std::invalid_argument("gauge unit cannot be empty");
          for (double sample : series.samples) {
            if (!std::isfinite(sample)) {
              throw std::invalid_argument("gauge samples must be finite");
            }
          }
        }
      },
      metric);
}

[[nodiscard]] inline std::string EscapeJson(std::string_view text)
{
  std::string escaped;
  escaped.reserve(text.size());
  for (char character : text) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          escaped += fmt::format("\\u{:04x}", static_cast<unsigned char>(character));
        } else {
          escaped += character;
        }
    }
  }
  return escaped;
}

}  // namespace detail

struct BenchmarkRun {
  RunInfo info;
  WorkloadInfo workload;
  std::optional<VerificationResult> verification;
  std::vector<ResourceProfile> profiles;
  std::vector<AlgorithmStage> stages;

  [[nodiscard]] std::string TypeTag() const
  {
    if (info.index_type.size() < 4 || info.offset_type.size() < 4) return "-";
    return "i" + info.index_type.substr(3) + "/o" + info.offset_type.substr(3);
  }

  [[nodiscard]] const SecondsSeries& WallTime() const
  {
    for (const ResourceProfile& profile : profiles) {
      if (!std::holds_alternative<RunScope>(profile.scope)) continue;
      for (const MetricSeries& metric : profile.metrics) {
        if (const auto* seconds = std::get_if<SecondsSeries>(&metric);
            seconds != nullptr && seconds->name == metric::kWallTime) {
          return *seconds;
        }
      }
    }
    throw std::invalid_argument("benchmark run requires run-scoped wall_time samples");
  }

  [[nodiscard]] TimingStats Stats() const { return TimingStats::From(WallTime().samples); }

  [[nodiscard]] double Gflops() const
  {
    const double median = Stats().median;
    return median > 0.0 ? workload.flops_per_iteration / median * 1.0e-9 : 0.0;
  }

  [[nodiscard]] double GbytesPerSecond() const
  {
    const double median = Stats().median;
    return median > 0.0 ? workload.bytes_per_iteration / median * 1.0e-9 : 0.0;
  }
};

inline void ValidateProfiles(const std::vector<ResourceProfile>& profiles)
{
  std::vector<std::string> scope_keys;
  for (const ResourceProfile& profile : profiles) {
    detail::ValidateScope(profile.scope);
    const std::string scope_key = detail::ScopeKey(profile.scope);
    if (std::find(scope_keys.begin(), scope_keys.end(), scope_key) != scope_keys.end()) {
      throw std::invalid_argument("a benchmark run cannot contain duplicate profile scopes");
    }
    scope_keys.push_back(scope_key);

    std::vector<std::string> metric_names;
    for (const MetricSeries& metric : profile.metrics) {
      detail::ValidateMetric(metric);
      const std::string& name =
          std::visit([](const auto& series) -> const std::string& { return series.name; }, metric);
      if (std::find(metric_names.begin(), metric_names.end(), name) != metric_names.end()) {
        throw std::invalid_argument("a profile cannot contain duplicate metric names");
      }
      metric_names.push_back(name);
    }
  }
}

inline void ValidateBenchmarkRun(const BenchmarkRun& run)
{
  if (run.info.kernel.empty() || run.info.backend.empty() || run.info.dataset.empty()) {
    throw std::invalid_argument("kernel, backend and dataset cannot be empty");
  }
  if (run.info.thread_count.has_value() && *run.info.thread_count < 1) {
    throw std::invalid_argument("thread count must be positive when recorded");
  }
  if (run.workload.rows < 0 || run.workload.columns < 0 || run.workload.input_nonzeros < 0 ||
      run.workload.flops_per_iteration < 0.0 || run.workload.bytes_per_iteration < 0.0) {
    throw std::invalid_argument("workload sizes and costs cannot be negative");
  }
  if (run.workload.output_nonzeros.has_value() && *run.workload.output_nonzeros < 0) {
    throw std::invalid_argument("output nonzeros cannot be negative");
  }
  if (run.verification.has_value() &&
      (!std::isfinite(run.verification->maximum_error) || run.verification->maximum_error < 0.0)) {
    throw std::invalid_argument("verification error must be finite and nonnegative");
  }

  ValidateProfiles(run.profiles);
  std::vector<std::string> stage_names;
  for (const AlgorithmStage& stage : run.stages) {
    if (stage.name.empty()) throw std::invalid_argument("algorithm stage name cannot be empty");
    if (std::find(stage_names.begin(), stage_names.end(), stage.name) != stage_names.end()) {
      throw std::invalid_argument("a benchmark run cannot contain duplicate stage names");
    }
    stage_names.push_back(stage.name);
    ValidateProfiles(stage.profiles);
  }
  (void)run.WallTime();
}

class BenchmarkReport
{
 public:
  explicit BenchmarkReport(ReportInfo info) : info_(std::move(info))
  {
    if (info_.title.empty()) throw std::invalid_argument("report title cannot be empty");
  }

  void AddRun(BenchmarkRun run)
  {
    ValidateBenchmarkRun(run);
    runs_.push_back(std::move(run));
  }

  [[nodiscard]] const ReportInfo& info() const { return info_; }

  [[nodiscard]] const std::vector<BenchmarkRun>& runs() const { return runs_; }

  void Validate() const
  {
    if (info_.title.empty()) throw std::invalid_argument("report title cannot be empty");
    for (const BenchmarkRun& run : runs_) ValidateBenchmarkRun(run);
  }

 private:
  ReportInfo info_;
  std::vector<BenchmarkRun> runs_;
};

namespace detail
{

struct RenderedMetric {
  std::string name;
  std::string unit;
  const char* summary;
  std::size_t samples;
  double headline;
  double minimum;
  double maximum;
};

[[nodiscard]] inline RenderedMetric RenderMetric(const MetricSeries& metric)
{
  return std::visit(
      [](const auto& series) {
        using Series = std::decay_t<decltype(series)>;
        std::vector<double> values;
        values.reserve(series.samples.size());
        for (auto sample : series.samples) values.push_back(static_cast<double>(sample));
        constexpr bool kAllowNegative = std::is_same_v<Series, GaugeSeries>;
        const TimingStats stats = TimingStats::From(values, kAllowNegative);
        double scale = 1.0;
        std::string unit;
        if constexpr (std::is_same_v<Series, SecondsSeries>) {
          scale = 1.0e3;
          unit = "ms";
        } else if constexpr (std::is_same_v<Series, BytesSeries>) {
          scale = 1.0 / (1024.0 * 1024.0);
          unit = "MiB";
        } else if constexpr (std::is_same_v<Series, CountSeries>) {
          unit = "count";
        } else if constexpr (std::is_same_v<Series, RatioSeries>) {
          scale = 100.0;
          unit = "%";
        } else {
          unit = series.unit;
        }
        return RenderedMetric{series.name,
                              std::move(unit),
                              SummaryName(series.headline),
                              series.samples.size(),
                              Headline(series.samples, series.headline, kAllowNegative) * scale,
                              stats.minimum * scale,
                              stats.maximum * scale};
      },
      metric);
}

[[nodiscard]] inline std::string Truncate(std::string_view text, std::size_t limit)
{
  return text.size() <= limit ? std::string(text) : std::string(text.substr(0, limit - 1)) + "~";
}

}  // namespace detail

class TextReportRenderer
{
 public:
  [[nodiscard]] std::string RenderSummary(const BenchmarkReport& report) const
  {
    report.Validate();
    constexpr int kWidth = 145;
    std::string output;
    output += fmt::format("{:=^{}}\n{:^{}}\n{:=^{}}\n", "", kWidth, report.info().title, kWidth, "",
                          kWidth);
    for (const auto& [key, value] : report.info().metadata) {
      output += fmt::format("{:<20}: {}\n", key, value);
    }

    output += fmt::format("{:-^{}}\n", "", kWidth);
    output +=
        fmt::format("{:<18}{:<10}{:<24}{:>10}{:>6}{:>8}{:>7}{:>12}{:>12}{:>7}{:>10}{:>10}{:>10}\n",
                    "backend", "kernel", "dataset", "idx/off", "prec", "threads", "iters",
                    "median_ms", "min_ms", "cv_%", "GFLOP/s", "GB/s", "max_err");
    output += fmt::format("{:-^{}}\n", "", kWidth);

    for (const BenchmarkRun& run : report.runs()) {
      const TimingStats stats = run.Stats();
      const std::string error = run.verification.has_value()
                                    ? fmt::format("{:.1e}", run.verification->maximum_error)
                                    : "-";
      output += fmt::format(
          "{:<18}{:<10}{:<24}{:>10}{:>6}{:>8}{:>7}{:>12.4f}{:>12.4f}{:>7.2f}{:>10.2f}{:>10.2f}{:>"
          "10}\n",
          detail::Truncate(run.info.backend, 17), detail::Truncate(run.info.kernel, 9),
          detail::Truncate(run.info.dataset, 23), run.TypeTag(), run.info.precision,
          run.info.thread_count.has_value() ? std::to_string(*run.info.thread_count) : "-",
          run.WallTime().samples.size(), stats.median * 1.0e3, stats.minimum * 1.0e3,
          stats.CoefficientOfVariation(), run.Gflops(), run.GbytesPerSecond(), error);
    }
    output += fmt::format("{:-^{}}\n", "", kWidth);
    return output;
  }

  [[nodiscard]] std::string RenderDetails(const BenchmarkReport& report) const
  {
    report.Validate();
    std::string output = "PROFILE DETAILS\n";
    for (const BenchmarkRun& run : report.runs()) {
      output +=
          fmt::format("\n{} / {} / {}\n", run.info.backend, run.info.kernel, run.info.dataset);
      for (const auto& [key, value] : run.info.parameters) {
        output += fmt::format("  parameter {}={}\n", key, value);
      }
      AppendProfiles(output, run.profiles, "run profiles");
      for (const AlgorithmStage& stage : run.stages) {
        output += fmt::format("  stage {}", stage.name);
        for (const auto& [key, value] : stage.attributes) {
          output += fmt::format(" {}={}", key, value);
        }
        output += "\n";
        AppendProfiles(output, stage.profiles, "stage profiles");
      }
    }
    return output;
  }

  [[nodiscard]] std::string Render(const BenchmarkReport& report) const
  {
    return RenderSummary(report) + "\n" + RenderDetails(report);
  }

 private:
  static void AppendProfiles(std::string& output, const std::vector<ResourceProfile>& profiles,
                             std::string_view label)
  {
    output += fmt::format("  {}\n", label);
    output += fmt::format("  {:<28}{:<30}{:>8}{:>10}{:>14}{:>14}{:>14}{:>10}\n", "scope", "metric",
                          "samples", "summary", "headline", "minimum", "maximum", "unit");
    for (const ResourceProfile& profile : profiles) {
      const std::string scope = detail::ScopeKey(profile.scope);
      if (!profile.attributes.empty()) {
        output += fmt::format("  {} attributes:", scope);
        for (const auto& [key, value] : profile.attributes) {
          output += fmt::format(" {}={}", key, value);
        }
        output += "\n";
      }
      for (const MetricSeries& metric : profile.metrics) {
        const detail::RenderedMetric rendered = detail::RenderMetric(metric);
        output += fmt::format("  {:<28}{:<30}{:>8}{:>10}{:>14.4f}{:>14.4f}{:>14.4f}{:>10}\n", scope,
                              rendered.name, rendered.samples, rendered.summary, rendered.headline,
                              rendered.minimum, rendered.maximum, rendered.unit);
      }
    }
  }
};

class JsonReportRenderer
{
 public:
  [[nodiscard]] std::string Render(const BenchmarkReport& report) const
  {
    report.Validate();
    std::string output =
        fmt::format("{{\n  \"schema_version\": 3,\n  \"title\": \"{}\",\n  \"info\": {{",
                    detail::EscapeJson(report.info().title));
    for (std::size_t i = 0; i < report.info().metadata.size(); ++i) {
      const auto& [key, value] = report.info().metadata[i];
      output += fmt::format("{}\n    \"{}\": \"{}\"", i == 0 ? "" : ",", detail::EscapeJson(key),
                            detail::EscapeJson(value));
    }
    output += report.info().metadata.empty() ? "}" : "\n  }";
    output += ",\n  \"runs\": [";

    for (std::size_t i = 0; i < report.runs().size(); ++i) {
      const BenchmarkRun& run = report.runs()[i];
      const TimingStats stats = run.Stats();
      const double verification_error =
          run.verification.has_value() ? run.verification->maximum_error : -1.0;
      output += fmt::format(
          "{}\n    {{\n"
          "      \"kernel\": \"{}\",\n      \"backend\": \"{}\",\n"
          "      \"dataset\": \"{}\",\n      \"precision\": \"{}\",\n"
          "      \"index_type\": \"{}\",\n      \"offset_type\": \"{}\",\n"
          "      \"threads\": {},\n      \"rows\": {},\n      \"columns\": {},\n"
          "      \"nonzeros\": {},\n      \"flops_per_iteration\": {},\n"
          "      \"bytes_per_iteration\": {},\n      \"verification_error\": {},\n"
          "      \"iterations\": {},\n      \"median_seconds\": {},\n"
          "      \"min_seconds\": {},\n      \"mean_seconds\": {},\n"
          "      \"stddev_seconds\": {},\n      \"gflops\": {},\n"
          "      \"gbytes_per_second\": {},\n      \"seconds\": [",
          i == 0 ? "" : ",", detail::EscapeJson(run.info.kernel),
          detail::EscapeJson(run.info.backend), detail::EscapeJson(run.info.dataset),
          detail::EscapeJson(run.info.precision), detail::EscapeJson(run.info.index_type),
          detail::EscapeJson(run.info.offset_type), run.info.thread_count.value_or(0),
          run.workload.rows, run.workload.columns, run.workload.input_nonzeros,
          run.workload.flops_per_iteration, run.workload.bytes_per_iteration, verification_error,
          run.WallTime().samples.size(), stats.median, stats.minimum, stats.mean, stats.stddev,
          run.Gflops(), run.GbytesPerSecond());
      for (std::size_t sample = 0; sample < run.WallTime().samples.size(); ++sample) {
        output += fmt::format("{}{}", sample == 0 ? "" : ", ", run.WallTime().samples[sample]);
      }
      output += "],\n      \"profiles\": ";
      AppendProfiles(output, run.profiles, 6);
      output += ",\n      \"stages\": [";
      for (std::size_t stage_index = 0; stage_index < run.stages.size(); ++stage_index) {
        const AlgorithmStage& stage = run.stages[stage_index];
        output += fmt::format("{}\n        {{\"name\": \"{}\", \"attributes\": {{",
                              stage_index == 0 ? "" : ",", detail::EscapeJson(stage.name));
        for (std::size_t attribute = 0; attribute < stage.attributes.size(); ++attribute) {
          const auto& [key, value] = stage.attributes[attribute];
          output += fmt::format("{}\"{}\": \"{}\"", attribute == 0 ? "" : ", ",
                                detail::EscapeJson(key), detail::EscapeJson(value));
        }
        output += "}, \"profiles\": ";
        AppendProfiles(output, stage.profiles, 8);
        output += "}";
      }
      output += run.stages.empty() ? "]\n    }" : "\n      ]\n    }";
    }
    output += report.runs().empty() ? "]\n}\n" : "\n  ]\n}\n";
    return output;
  }

 private:
  static void AppendProfiles(std::string& output, const std::vector<ResourceProfile>& profiles,
                             int indent)
  {
    output += "[";
    const std::string profile_indent(static_cast<std::size_t>(indent), ' ');
    const std::string field_indent(static_cast<std::size_t>(indent + 2), ' ');
    const std::string metric_indent(static_cast<std::size_t>(indent + 4), ' ');
    for (std::size_t profile_index = 0; profile_index < profiles.size(); ++profile_index) {
      const ResourceProfile& profile = profiles[profile_index];
      output += fmt::format("{}\n{}{{\"scope\": \"{}\", \"attributes\": {{",
                            profile_index == 0 ? "" : ",", profile_indent,
                            detail::EscapeJson(detail::ScopeKey(profile.scope)));
      for (std::size_t attribute = 0; attribute < profile.attributes.size(); ++attribute) {
        const auto& [key, value] = profile.attributes[attribute];
        output += fmt::format("{}\"{}\": \"{}\"", attribute == 0 ? "" : ", ",
                              detail::EscapeJson(key), detail::EscapeJson(value));
      }
      output += "}, \"metrics\": [";
      for (std::size_t metric_index = 0; metric_index < profile.metrics.size(); ++metric_index) {
        std::visit(
            [&](const auto& series) {
              using Series = std::decay_t<decltype(series)>;
              std::string unit;
              if constexpr (std::is_same_v<Series, SecondsSeries>) {
                unit = "seconds";
              } else if constexpr (std::is_same_v<Series, BytesSeries>) {
                unit = "bytes";
              } else if constexpr (std::is_same_v<Series, CountSeries>) {
                unit = "count";
              } else if constexpr (std::is_same_v<Series, RatioSeries>) {
                unit = "ratio";
              } else {
                unit = series.unit;
              }
              output += fmt::format(
                  "{}\n{}{{\"name\": \"{}\", \"unit\": \"{}\", \"headline\": \"{}\", "
                  "\"samples\": [",
                  metric_index == 0 ? "" : ",", metric_indent, detail::EscapeJson(series.name),
                  detail::EscapeJson(unit), detail::SummaryName(series.headline));
              for (std::size_t sample = 0; sample < series.samples.size(); ++sample) {
                output += fmt::format("{}{}", sample == 0 ? "" : ", ", series.samples[sample]);
              }
              output += "]}";
            },
            profile.metrics[metric_index]);
      }
      if (!profile.metrics.empty()) output += fmt::format("\n{}", field_indent);
      output += "]}";
    }
    if (!profiles.empty()) output += fmt::format("\n{}", std::string(indent - 2, ' '));
    output += "]";
  }
};

inline void PrintReportToStderr(std::string_view text) { fmt::print(stderr, "{}", text); }

inline void WriteReportFile(const std::string& path, std::string_view text)
{
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open report output file: " + path);
  output << text;
  if (!output) throw std::runtime_error("failed while writing report output file: " + path);
}

struct HostTimingSamples {
  std::vector<double> wall_seconds;
  std::vector<double> process_cpu_seconds;
};

[[nodiscard]] inline double ProcessCpuSeconds()
{
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage could not read process CPU time");
  }
  const double user = static_cast<double>(usage.ru_utime.tv_sec) +
                      static_cast<double>(usage.ru_utime.tv_usec) * 1.0e-6;
  const double system = static_cast<double>(usage.ru_stime.tv_sec) +
                        static_cast<double>(usage.ru_stime.tv_usec) * 1.0e-6;
  return user + system;
}

[[nodiscard]] inline double ThreadCpuSeconds()
{
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
    throw std::runtime_error("clock_gettime could not read thread CPU time");
  }
  return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1.0e-9;
}

/** @brief Collect paired wall and process CPU samples for each invocation. */
template <class Body>
[[nodiscard]] HostTimingSamples TimeHostIterations(Body&& body, int iterations)
{
  if (iterations < 1) {
    throw std::invalid_argument("TimeHostIterations requires at least one iteration");
  }

  HostTimingSamples samples;
  samples.wall_seconds.reserve(static_cast<std::size_t>(iterations));
  samples.process_cpu_seconds.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const double cpu_start = ProcessCpuSeconds();
    const auto wall_start = std::chrono::steady_clock::now();
    body();
    const auto wall_end = std::chrono::steady_clock::now();
    const double cpu_end = ProcessCpuSeconds();
    samples.wall_seconds.push_back(std::chrono::duration<double>(wall_end - wall_start).count());
    samples.process_cpu_seconds.push_back(cpu_end - cpu_start);
  }
  return samples;
}

/**
 * @brief Time `body` repeatedly, returning one wall-clock sample per iteration.
 *
 * GPU bodies must synchronize before returning so that asynchronous work is included.
 */
template <class Body>
[[nodiscard]] std::vector<double> TimeIterations(Body&& body, int iterations)
{
  if (iterations < 1) {
    throw std::invalid_argument("TimeIterations requires at least one iteration");
  }

  std::vector<double> seconds;
  seconds.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    body();
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    seconds.push_back(elapsed.count());
  }
  return seconds;
}

}  // namespace spcraft
