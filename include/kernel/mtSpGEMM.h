#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/CooMatrix.h"
#include "core/DcscMatrix.h"
#include "kernel/SpGEMMHash.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

/**
 * @brief Hash-based SpGEMM Algorithm with Dcsc format input and COO output.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const DcscMatrix<IT, NT, OT>& A,
                                                  const DcscMatrix<IT, NT, OT>& B);

/**
 * @brief Hash-based SpGEMM Algorithm with Csc format input and COO output.
 */
template <class SemiRing, class IT, class NT, class OT>
[[nodiscard]] CooMatrix<IT, NT, OT> OmpHashSpGEMM(const CscMatrix<IT, NT, OT>& A,
                                                  const CscMatrix<IT, NT, OT>& B);

}  // namespace spcraft

#include "HashSpGEMMCsc_impl.h"
#include "HashSpGEMMDcsc_impl.h"