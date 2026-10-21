#pragma once

#include "mpi/CommGrid3D.h"
#include "mpi/DistSpCOO.h"
#include "mpi/DistSpCSC.h"
#include "mpi/DistSpDCSC.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

/**
 * @brief Synchronous layered 3D SUMMA with sparse semiring reduction to layer zero.
 * Collective on grid.World(). A and B share grid.Plane(), compatible global
 * inner dimensions/block extents, and identical shapes across all layers.
 * Inner block k of A and B is stored only on layer k % grid.Layers(), using
 * the plane's full packed 2D indices. CommGrid3D::AOwner/BOwner describe ownership.
 * Each layer multiplies its assigned inner blocks with blocking 2D broadcasts.
 * Sorted partial COO products are reduced along fibers through SemiRing::Add.
 * Layer zero returns the reduced product sharing the input plane grid. Other
 * layers return a default-constructed empty matrix with no grid ownership.
 * Check grid.Layer() == 0 to identify result owners, including empty products.
 * The reduced result retains structural zeros.
 * Link SPCraft for PlusTimesRing<float/double> and signed (IT, OT) widths
 * (32, 32), (32, 64), or (64, 64). All grid owners must die before MPI_Finalize.
 */
template <class SemiRing, class IT, class NT, class OT>
SPND DistSpCOO<IT, NT, OT> Mpi3DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B, const CommGrid3D& g);

//! DCSC input overload with the same ownership, shape, and collective requirements.
template <class SemiRing, class IT, class NT, class OT>
SPND DISTSPCOO Mpi3DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B, const CommGrid3D& g);

}  // namespace spcraft
