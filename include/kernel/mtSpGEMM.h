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
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Hash-based SpGEMM Algorithm with Csc format input and COO output.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] SPCOO OmpHashSpGEMM(const SPCSC& A, const SPCSC& B);

/**
 * @brief Hash-based SpGEMM Algorithm with Dcsc format input and COO output.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] SPCOO OmpHashSpGEMM(const SPDCSC& A, const SPDCSC& B);

/**
 * @brief Hash-based SpGEMM Algorithm with Csr format input and COO output.
 * Nonempty CSR multiplication is not yet implemented.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] SPCOO OmpHashSpGEMM(const SPCSR& A, const SPCSR& B);

}  // namespace spcraft
