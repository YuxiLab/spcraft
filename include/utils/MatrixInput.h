#pragma once

#include <optional>
#include <string>

#include "utils/MatrixInputOptions.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

template <class IT, class NT, class OT>
class CooMatrix;

struct MatrixInput {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::optional<MatrixInputOptions> options;
  int exit_code = 0;
};

// Link SPCraftInput to use these compiled utilities. Keeping parsing out of
// kernel translation units lets them rebuild without recompiling the readers.
MatrixInput ParseMatrixInput(int argc, const char* const* argv, const char* description);

// The compiled reader supports the same six type combinations as the examples.
template <class IT, class NT, class OT = IT>
CooMatrix<IT, NT, OT> ReadMatrixInput(const std::string& path);

}  // namespace spcraft
