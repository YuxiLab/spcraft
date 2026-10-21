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
template <class SemiRing, class IT, class NT, class OT>
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B);

/**
 * @brief Synchronous 2D SUMMA SpGEMM Algorithm (SPDCSC format).
 */
template <class SemiRing, class IT, class NT, class OT>
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B);

}  // namespace spcraft
