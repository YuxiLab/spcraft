#pragma once

#include <mpi.h>

#include "core/CooMatrix.h"
#include "core/DcscMatrix.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Collective 2D SUMMA SpGEMM on local DCSC blocks, returning the local COO block.
 * A and B use compatible uniform block partitions on a square 2D Cartesian MPI grid.
 * The grid coordinates are (block row, block column). Input and output indices
 * are local to each block; the output has A.m rows and B.n columns.
 * Link SPCraft with MPI enabled for PlusTimesRing<float> or PlusTimesRing<double>
 * and signed (IT, OT) widths (32, 32), (32, 64), or (64, 64).
 *
 * Student exercise: the current stub returns an empty block. Implement SUMMA
 * broadcasts, local products, and accumulation to pass the correctness tests.
 */
SP_SR_MAT_TEMP
SPND SPCOO Mpi2DSummaSpGEMM(const SPDCSC& A, const SPDCSC& B, MPI_Comm grid);

}  // namespace spcraft
