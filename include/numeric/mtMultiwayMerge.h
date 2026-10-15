#pragma once

#include <vector>

#include "core/CooMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Merge column-then-row sorted COO matrices with matching dimensions.
 * Inputs remain unchanged; the returned matrix owns independent storage.
 * Duplicate coordinates, including duplicates within an input, are combined
 * through SemiRing::Add. Structural zeros are retained, and output coordinates
 * are sorted and unique. An empty input vector produces an empty m-by-n matrix.
 * Column splits run in parallel when OpenMP is available, otherwise serially.
 * Link SPCraft for PlusTimesRing<float/double/bool> with signed (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64).
 * @throws std::invalid_argument If dimensions are negative or do not match.
 * @throws std::overflow_error If the merged nnz does not fit OT.
 * @see https://github.com/PASSIONLab/CombBLAS/blob/master/include/CombBLAS/MultiwayMerge.h
 */
SP_SR_MAT_TEMP
SPND SPCOO OmpMultiwayMerge(const std::vector<SPCOO>& inputs, IT m, IT n);

}  // namespace spcraft
