#include "utils/MatrixInput.h"

#include <filesystem>

#include <cxxopts.hpp>
#include <fmt/format.h>

#include "core/CooMatrix.h"

namespace spcraft
{
MatrixInput ParseMatrixInput(int argc, char** argv, const char* description)
{
  cxxopts::Options options(argv[0], description);
  // clang-format off
  options.positional_help("<matrix.mtx>");
  options.add_options()
  ("matrix", "Matrix Market input file", cxxopts::value<std::string>())
  ("h,help", "Print usage");
  options.parse_positional({"matrix"});
  // clang-format on

  const auto arguments = options.parse(argc, argv);
  if (arguments.count("help") != 0 || arguments.count("matrix") == 0) {
    fmt::print("{}\n", options.help());
    return {std::nullopt, arguments.count("help") != 0 ? 0 : 1};
  }
  const auto path = arguments["matrix"].as<std::string>();
  if (!std::filesystem::exists(path)) {
    fmt::print(stderr, "Matrix file does not exist: {}\n", path);
    return {std::nullopt, 1};
  }
  return {path, 0};
}

spcraft::CooMatrix<std::int64_t, double, std::int64_t> ReadMatrixInput(const std::string& path)
{
  return spcraft::CooMatrix<std::int64_t, double, std::int64_t>(path);
}
}  // namespace spcraft
