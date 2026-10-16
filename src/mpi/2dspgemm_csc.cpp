#include "mpi/2dspgemm.h"
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "kernel/mtSpGEMM.h"
#include "mpi/CommGrid2D.h"
#include "mpi/mpiutil.h"
#include "numeric/mtMultiwayMerge.h"
#include "semiring/SemiRing.h"

namespace spcraft
{

namespace
{

template <class IT, class OT>
struct BlockInfo {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT m = 0;    //!< Local submatrix rows.
  IT n = 0;    //!< Local submatrix columns.
  OT nnz = 0;  //!< Local submatrix nonzeros.
};

template <class IT, class OT>
std::vector<BlockInfo<IT, OT>> GatherLocalMatInfo(const MPI_Comm mpi_comm, IT m, IT n, OT nnz)
{
  int world_size;
  MPI_Comm_size(mpi_comm, &world_size);
  std::vector<BlockInfo<IT, OT>> res(world_size);

  const BlockInfo<IT, OT> local{m, n, nnz};
  const int lengths[] = {1, 1, 1};
  const MPI_Datatype types[] = {MPIType<IT>(), MPIType<IT>(), MPIType<OT>()};

  MPI_Aint base;
  MPI_Aint offsets[3];
  MPI_Get_address(&local, &base);
  MPI_Get_address(&local.m, &offsets[0]);
  MPI_Get_address(&local.n, &offsets[1]);
  MPI_Get_address(&local.nnz, &offsets[2]);
  for (MPI_Aint& offset : offsets) {
    offset -= base;
  }

  MPI_Datatype raw_type;
  MPI_Datatype block_type;
  MPI_Type_create_struct(3, lengths, offsets, types, &raw_type);
  // Match the vector element stride, including any trailing padding.
  MPI_Type_create_resized(raw_type, 0, sizeof(BlockInfo<IT, OT>), &block_type);
  MPI_Type_commit(&block_type);
  MPI_Type_free(&raw_type);

  MPI_Allgather(&local, 1, block_type, res.data(), 1, block_type, mpi_comm);

  MPI_Type_free(&block_type);
  return res;
}

}  // namespace

SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B)
{
  // @profiler-region dimchk start
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>, "Semiring and Matrix type mismatch");
  // Each rank owns one contiguous submatrix on the shared process grid.
  const bool same_grid = A.commgrid && B.commgrid && A.commgrid == B.commgrid;
  if (!same_grid || A.commgrid->World() == MPI_COMM_NULL) {
    throw std::invalid_argument("SUMMA inputs must share the same live communicator grid");
  }
  const CommGrid2D& grid = *A.commgrid;
  if (grid.Rows() != grid.Cols()) {
    throw std::invalid_argument("SUMMA currently requires a square process grid");
  }
  if (A.n != B.m) {
    throw std::invalid_argument("SUMMA inputs need compatible inner dimensions");
  }
  const bool a_shape_matches = A.local.m == A.LocalRows() && A.local.n == A.LocalCols();
  const bool b_shape_matches = B.local.m == B.LocalRows() && B.local.n == B.LocalCols();
  if (!a_shape_matches || !b_shape_matches) {
    throw std::invalid_argument("SUMMA local dimensions do not match the process-grid partition");
  }
  DISTSPCOO result(A.commgrid, A.m, B.n);
  if (A.n == 0) return result;
  // @profiler-region dimchk end

  // A metadata follows process columns; B metadata follows process rows.
  using Block = BlockInfo<IT, OT>;
  const MPI_Comm row_world = grid.RowWorld();
  const MPI_Comm col_world = grid.ColWorld();
  const std::vector<Block> A_info = GatherLocalMatInfo(row_world, A.local.m, A.local.n, A.local.nnz);
  const std::vector<Block> B_info = GatherLocalMatInfo(col_world, B.local.m, B.local.n, B.local.nnz);
  std::vector<SPCOO> cstages;
  for (int q = 0; q < grid.Cols(); ++q) {
    // @profiler-iter q start

    // @profiler-region bcast_A start
    // bcast A
    const Block& a = A_info[q];
    SPCSC opA;
    if (q == grid.Col()) {
      // self, non-owning view
      opA = SPCSC(A.local.col_ptr, A.local.row_id, A.local.val, A.local.nnz, A.local.m, A.local.n);
    } else {
      // allocate memory
      opA.Allocate(a.nnz, a.m, a.n);
    }
    MPI_Bcast(opA.col_ptr, a.n + 1, MPIType<OT>(), q, row_world);
    MPI_Bcast(opA.row_id, a.nnz, MPIType<IT>(), q, row_world);
    MPI_Bcast(opA.val, a.nnz, MPIType<NT>(), q, row_world);
    // @profiler-region bcast_A end
    // @profiler-region bcast_B start
    // bcast B
    const Block& b = B_info[q];
    SPCSC opB;
    if (q == grid.Row()) {
      // self, non-owning view
      opB = SPCSC(B.local.col_ptr, B.local.row_id, B.local.val, B.local.nnz, B.local.m, B.local.n);
    } else {
      // allocate memory
      opB.Allocate(b.nnz, b.m, b.n);
    }
    MPI_Bcast(opB.col_ptr, b.n + 1, MPIType<OT>(), q, col_world);
    MPI_Bcast(opB.row_id, b.nnz, MPIType<IT>(), q, col_world);
    MPI_Bcast(opB.val, b.nnz, MPIType<NT>(), q, col_world);
    // @profiler-region bcast_B end
    // @profiler-region localspgemm start
    SPCOO tmpc = OmpHashSpGEMM<SemiRing>(opA, opB);
    cstages.push_back(std::move(tmpc));
    // @profiler-region localspgemm end
    // @profiler-iter q end
  }
  // @profiler-region multiwaymerge start
  // merge stages' COO
  result.local = OmpMultiwayMerge<SemiRing>(cstages, A.local.m, B.local.n);
  // @profiler-region multiwaymerge end
  return result;
}

#define SPCRAFT_DEFINE_SUMMA_2D_TYPES(IT, NT, OT)                                 \
  template DistSpCOO<IT, NT, OT> Mpi2DSummaSpGEMM<PlusTimesRing<NT>, IT, NT, OT>( \
      const DistSpCSC<IT, NT, OT>&, const DistSpCSC<IT, NT, OT>&);

#define SPCRAFT_DEFINE_SUMMA_2D(r, product)                                          \
  SPCRAFT_DEFINE_SUMMA_2D_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                BOOST_PP_SEQ_ELEM(1, product),                       \
                                BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Keep compiled storage pairs consistent with local SpGEMM.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SUMMA_2D,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_SUMMA_2D
#undef SPCRAFT_DEFINE_SUMMA_2D_TYPES

}  // namespace spcraft
