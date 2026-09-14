#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spcraft
{

/**
 * @brief Summary statistics over a set of per-iteration timings, in seconds.
 */
struct TimingStats {
  double minimum = 0.0;
  double median = 0.0;
  double mean = 0.0;
  double stddev = 0.0;

  //! Coefficient of variation in percent; the headline noise indicator.
  [[nodiscard]] double CoefficientOfVariation() const
  {
    return mean > 0.0 ? 100.0 * stddev / mean : 0.0;
  }

  [[nodiscard]] static TimingStats From(const std::vector<double>& seconds)
  {
    if (seconds.empty()) {
      throw std::invalid_argument("TimingStats requires at least one timing sample");
    }

    TimingStats stats;
    std::vector<double> sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    stats.minimum = sorted.front();
    const std::size_t middle = sorted.size() / 2;
    stats.median =
        sorted.size() % 2 == 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];
    stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                 static_cast<double>(sorted.size());

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

/**
 * @brief One measured configuration: a kernel run on one dataset at one setting.
 *
 * `seconds` holds the raw per-iteration timings rather than a summary, so the
 * JSON log supports distribution analysis that aggregates would discard.
 */
struct BenchmarkRun {
  std::string kernel;       //!< "SpMV", "SpGEMM", ...
  std::string backend;      //!< "SPCraft-OpenMP", "cuSPARSE", ...
  std::string dataset;      //!< Matrix name or generator description.
  std::string precision;    //!< "FP32" or "FP64".
  std::string index_type;   //!< "int32" or "int64"; empty when not recorded.
  std::string offset_type;  //!< "int32" or "int64"; empty when not recorded.
  int threads = 0;          //!< 0 when a thread count does not apply.
  std::int64_t rows = 0;
  std::int64_t columns = 0;
  std::int64_t nonzeros = 0;
  double flops_per_iteration = 0.0;
  double bytes_per_iteration = 0.0;
  //! Largest absolute deviation from the reference; negative when unverified.
  double verification_error = -1.0;
  std::vector<double> seconds;

  //! Compact "i32/o64" label; "-" when the driver did not record the types.
  [[nodiscard]] std::string TypeTag() const
  {
    if (index_type.empty() || offset_type.empty()) return "-";
    return "i" + index_type.substr(3) + "/o" + offset_type.substr(3);
  }

  [[nodiscard]] TimingStats Stats() const
  {
    return TimingStats::From(seconds);
  }
  [[nodiscard]] double Gflops() const
  {
    const double median = Stats().median;
    return median > 0.0 ? flops_per_iteration / median * 1.0e-9 : 0.0;
  }
  [[nodiscard]] double GbytesPerSecond() const
  {
    const double median = Stats().median;
    return median > 0.0 ? bytes_per_iteration / median * 1.0e-9 : 0.0;
  }
};

/**
 * @brief Time `body` `iterations` times, recording every individual iteration.
 *
 * Every SPCraft benchmark times through this function so that the reported
 * numbers are comparable across kernels and backends. Warm up before calling:
 * the first iteration is measured like any other.
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

/**
 * @brief Console report shared by every SPCraft benchmark, plus a raw JSON log.
 *
 * Rows print as they are added so a long sweep shows progress. WriteJson emits
 * every per-iteration timing for offline analysis in Python.
 */
class BenchmarkReport
{
 public:
  explicit BenchmarkReport(std::string title) : title_(std::move(title))
  {
  }

  //! Add a banner line, printed above the table.
  void AddInfo(std::string key, std::string value)
  {
    info_.emplace_back(std::move(key), std::move(value));
  }

  //! Print the title, banner lines, and column headings.
  void PrintHeader() const
  {
    fmt::print("{:=^{}}\n", "", kWidth);
    fmt::print("{:^{}}\n", title_, kWidth);
    fmt::print("{:=^{}}\n", "", kWidth);
    for (const auto& [key, value] : info_) {
      fmt::print("{:<16}: {}\n", key, value);
    }
    fmt::print("{:-^{}}\n", "", kWidth);
    fmt::print("{:<24}{:>10}{:>6}{:>8}{:>7}{:>12}{:>12}{:>7}{:>10}{:>10}{:>10}\n", "dataset",
               "idx/off", "prec", "threads", "iters", "median_ms", "min_ms", "cv_%", "GFLOP/s",
               "GB/s", "max_err");
    fmt::print("{:-^{}}\n", "", kWidth);
  }

  //! Record a run and print its row.
  void Add(BenchmarkRun run)
  {
    const TimingStats stats = run.Stats();
    fmt::print(
        "{:<24}{:>10}{:>6}{:>8}{:>7}{:>12.4f}{:>12.4f}{:>7.2f}{:>10.2f}{:>10.2f}{:>10}\n",
        Truncate(run.dataset, 23), run.TypeTag(), run.precision,
        run.threads > 0 ? std::to_string(run.threads) : "-", run.seconds.size(),
        stats.median * 1.0e3, stats.minimum * 1.0e3, stats.CoefficientOfVariation(),
        run.Gflops(), run.GbytesPerSecond(),
        run.verification_error < 0.0 ? std::string("-")
                                     : fmt::format("{:.1e}", run.verification_error));
    runs_.push_back(std::move(run));
  }

  void PrintFooter() const
  {
    fmt::print("{:-^{}}\n", "", kWidth);
  }

  [[nodiscard]] const std::vector<BenchmarkRun>& runs() const
  {
    return runs_;
  }

  /**
   * @brief Write every run, including raw per-iteration timings, as JSON.
   *
   * The schema is stable and consumed by the analysis scripts under scripts/.
   */
  void WriteJson(const std::string& path) const
  {
    std::ofstream out(path);
    if (!out) {
      throw std::runtime_error("cannot open report output file: " + path);
    }

    out << fmt::format("{{\n  \"title\": \"{}\",\n  \"info\": {{", Escape(title_));
    for (std::size_t i = 0; i < info_.size(); ++i) {
      out << fmt::format("{}\n    \"{}\": \"{}\"", i == 0 ? "" : ",", Escape(info_[i].first),
                         Escape(info_[i].second));
    }
    out << (info_.empty() ? "}" : "\n  }") << ",\n  \"runs\": [";

    for (std::size_t i = 0; i < runs_.size(); ++i) {
      const BenchmarkRun& run = runs_[i];
      const TimingStats stats = run.Stats();
      out << fmt::format(
          "{}\n    {{\n"
          "      \"kernel\": \"{}\",\n      \"backend\": \"{}\",\n"
          "      \"dataset\": \"{}\",\n      \"precision\": \"{}\",\n"
          "      \"index_type\": \"{}\",\n      \"offset_type\": \"{}\",\n"
          "      \"threads\": {},\n      \"rows\": {},\n      \"columns\": {},\n"
          "      \"nonzeros\": {},\n      \"flops_per_iteration\": {},\n"
          "      \"bytes_per_iteration\": {},\n      \"verification_error\": {},\n"
          "      \"iterations\": {},\n"
          "      \"median_seconds\": {},\n      \"min_seconds\": {},\n"
          "      \"mean_seconds\": {},\n      \"stddev_seconds\": {},\n"
          "      \"gflops\": {},\n      \"gbytes_per_second\": {},\n"
          "      \"seconds\": [",
          i == 0 ? "" : ",", Escape(run.kernel), Escape(run.backend), Escape(run.dataset),
          Escape(run.precision), Escape(run.index_type), Escape(run.offset_type), run.threads,
          run.rows, run.columns, run.nonzeros, run.flops_per_iteration,
          run.bytes_per_iteration, run.verification_error, run.seconds.size(), stats.median,
          stats.minimum, stats.mean, stats.stddev, run.Gflops(), run.GbytesPerSecond());

      for (std::size_t sample = 0; sample < run.seconds.size(); ++sample) {
        out << fmt::format("{}{}", sample == 0 ? "" : ", ", run.seconds[sample]);
      }
      out << "]\n    }";
    }
    out << (runs_.empty() ? "]" : "\n  ]") << "\n}\n";

    if (!out) {
      throw std::runtime_error("failed while writing report output file: " + path);
    }
  }

 private:
  static constexpr int kWidth = 116;

  [[nodiscard]] static std::string Truncate(const std::string& text, std::size_t limit)
  {
    return text.size() <= limit ? text : text.substr(0, limit - 1) + "~";
  }

  [[nodiscard]] static std::string Escape(const std::string& text)
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

  std::string title_;
  std::vector<std::pair<std::string, std::string>> info_;
  std::vector<BenchmarkRun> runs_;
};

}  // namespace spcraft
