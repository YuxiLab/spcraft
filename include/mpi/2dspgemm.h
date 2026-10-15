#pragma once

#include "mpi/DistSpCOO.h"
#include "mpi/DistSpCSC.h"
#include "mpi/DistSpDCSC.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Synchronous collective 2D SUMMA SpGEMM on distributed CSC matrices.
 * A and B must share the same live CommGrid2D object and compatible global
 * inner dimensions. Each rank owns one contiguous local submatrix. Choose the
 * process grid first and derive row/column block extents from global dimensions.
 * The grid must be square. All ranks supply consistent shapes and call collectively.
 * The returned COO shares the grid and has global shape (A.m, B.n).
 * Local m, n, and nnz are gathered into metadata tables indexed by process row
 * and column before multiplication.
 * At stage q, A's process-column-q block is broadcast along each process row,
 * and B's process-row-q block is broadcast along each process column.
 * Owners borrow their local CSC arrays; receivers allocate their stage buffers.
 * These matrices feed the local CSC hash SpGEMM kernel.
 * Output entries are sorted by column then row, retaining structural zeros.
 * Link SPCraft for PlusTimesRing<float/double> and signed (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64).
 */
SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B);

//! DCSC overload: square grid, matching inner block extents, local DCSC hash SpGEMM.
SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B);

}  // namespace spcraft
