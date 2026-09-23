#pragma once

// Opt in with target_precompile_headers(target PRIVATE <utils/StablePch.h>).
// Cache stable storage and third-party headers, but never kernels or SpCraft.h:
// editing a kernel must recompile the driver without rebuilding this PCH.
#include <fmt/format.h>

#include "core/CooMatrix.h"
#include "core/DcscMatrix.h"
#include "core/DenseVector.h"

#ifdef SPCRAFT_USE_MKL
#include <mkl.h>
#endif
