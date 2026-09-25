#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "utils/MatrixMacros.h"
namespace spcraft
{
SP_MAT_TEMP
class CooMatrix;

struct MatrixInput {
  std::optional<std::string> path;
  int exit_code;
};

// Link SPCraftInput to use these compiled utilities. Keeping parsing out of
// kernel translation units lets them rebuild without recompiling the readers.
MatrixInput ParseMatrixInput(int argc, char** argv, const char* description);
// The compiled reader uses 64-bit indices/offsets and double-precision values.
spcraft::CooMatrix<std::int64_t, double, std::int64_t> ReadMatrixInput(const std::string& path);
}  // namespace spcraft
