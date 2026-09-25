#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "utils/utils.h"

namespace spcraft
{

/**
 * @brief Hash-based SpGEMM Algorithm with Dcsc format input and COO output.
 */
SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPDCSC& A, const SPDCSC& B);

/**
 * @brief Hash-based SpGEMM Algorithm with Csc format input and COO output.
 */
SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPCSC& A, const SPCSC& B);

/**
 * @brief Hash-based SpGEMM Algorithm with Csc format input and COO output.
 */
SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPCSR& A, const SPCSR& B);

}  // namespace spcraft

#include "HashSpGEMMCsr_impl.h"
#include "HashSpGEMMCsc_impl.h"
#include "HashSpGEMMDcsc_impl.h"
