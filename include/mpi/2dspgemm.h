#pragma once

#include "mpi/DistSpCOO.h"
#include "mpi/DistSpCSC.h"
#include "mpi/DistSpDCSC.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Synchronous 2D SUMMA SpGEMM Algorithm (SPCSC format).
 */
SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B);

/**
 * @brief Synchronous 2D SUMMA SpGEMM Algorithm (SPDCSC format).
 */
SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B);

}  // namespace spcraft
