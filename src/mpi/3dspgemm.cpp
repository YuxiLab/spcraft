#include "mpi/3dspgemm.h"

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "core/AllocationGuard.h"
#include "mpi/DistSpCOO.h"
#include "mpi/mpiutil.h"
#include "semiring/SemiRing.h"
#include "SummaCommon.h"

namespace spcraft
{
namespace
{

//! Chunked point-to-point transfer uses the grid's isolated fiber communicator.
template <class T>
void SendArray(const T* data, int64_t count, MPI_Comm comm)
{
  constexpr int64_t kMaxCount = std::numeric_limits<int>::max();
  for (int64_t start = 0; start < count;) {
    const int chunk = static_cast<int>(std::min(count - start, kMaxCount));
    MPI_Send(data + start, chunk, MPIType<T>(), 0, 0, comm);
    start += chunk;
  }
}

template <class T>
void ReceiveArray(T* data, int64_t count, int source, MPI_Comm comm)
{
  constexpr int64_t kMaxCount = std::numeric_limits<int>::max();
  for (int64_t start = 0; start < count;) {
    const int chunk = static_cast<int>(std::min(count - start, kMaxCount));
    MPI_Recv(data + start, chunk, MPIType<T>(), source, 0, comm, MPI_STATUS_IGNORE);
    start += chunk;
  }
}

//! Communicate native fields, never struct padding or an MPI byte count narrowed from nnz.
template <class SemiRing, class IT, class NT, class OT, class Matrix>
DISTSPCOO RunLayeredSumma(const Matrix& A, const Matrix& B, const CommGrid3D& grid)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>);
  if (grid.World() == MPI_COMM_NULL) {
    throw std::invalid_argument("3D SUMMA needs a live grid");
  }
  DistSpCOO<IT, NT, OT> partial;
  // Make shape errors collective before layers enter different broadcast schedules.
  int invalid = A.commgrid != grid.Plane() || B.commgrid != grid.Plane() ? 1 : 0;
  try {
    detail::CheckSummaInputs(A, B);
  } catch (const std::invalid_argument&) {
    invalid = 1;
  }
  int any_invalid = 0;
  MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, grid.World());
  if (any_invalid) throw std::invalid_argument("3D SUMMA inputs have incompatible partitions");
  partial = detail::RunSummaStages<SemiRing, IT, NT, OT>(A, B, grid.Layer(), grid.Layers());

  // Gather sparse products in layer order. Integer and value buffers avoid ABI padding.
  const MPI_Comm fiber = grid.FiberWorld();
  const OT count = partial.local.nnz;
  std::vector<IT> rows;
  std::vector<IT> columns;
  std::vector<NT> values;
  if (grid.Layer() != 0) {
    const size_t indices = detail::CheckedElementCount<IT>(count);
    const size_t entries = detail::CheckedElementCount<NT>(count);
    rows.resize(indices);
    columns.resize(indices);
    values.resize(entries);
    for (OT p = 0; p < count; ++p) {
      rows[p] = partial.local.entries[p].row;
      columns[p] = partial.local.entries[p].col;
      values[p] = partial.local.entries[p].val;
    }
    MPI_Send(&count, 1, MPIType<OT>(), 0, 0, fiber);
    SendArray(rows.data(), count, fiber);
    SendArray(columns.data(), count, fiber);
    SendArray(values.data(), count, fiber);
    return {};
  }
  for (int layer = 1; layer < grid.Layers(); ++layer) {
    OT incoming = 0;
    MPI_Recv(&incoming, 1, MPIType<OT>(), layer, 0, fiber, MPI_STATUS_IGNORE);
    const size_t indices = detail::CheckedElementCount<IT>(incoming);
    const size_t entries = detail::CheckedElementCount<NT>(incoming);
    rows.resize(indices);
    columns.resize(indices);
    values.resize(entries);
    ReceiveArray(rows.data(), incoming, layer, fiber);
    ReceiveArray(columns.data(), incoming, layer, fiber);
    ReceiveArray(values.data(), incoming, layer, fiber);
    CooMatrix<IT, NT, OT> other;
    other.Allocate(incoming, partial.local.m, partial.local.n);
    for (OT p = 0; p < incoming; ++p) other.entries[p] = {rows[p], columns[p], values[p]};
    partial.local = detail::AddProducts<SemiRing>(partial.local, other);
  }
  return partial;
}

}  // namespace

SP_SR_MAT_TEMP
SPND DistSpCOO<IT, NT, OT> Mpi3DSummaSpGEMM(const DISTSPCSC& A, const DISTSPCSC& B, const CommGrid3D& g)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>);
  DISTSPCOO result = RunLayeredSumma<SemiRing, IT, NT, OT>(A, B, g);
  return result;
}

SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi3DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B, const CommGrid3D& g)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>);
  DISTSPCOO result = RunLayeredSumma<SemiRing, IT, NT, OT>(A, B, g);
  return result;
}

#define SPCRAFT_DEFINE_SUMMA_3D_TYPES(IT, NT, OT)                                     \
  template DistSpCOO<IT, NT, OT> Mpi3DSummaSpGEMM<PlusTimesRing<NT>, IT, NT, OT>(     \
      const DistSpCSC<IT, NT, OT>&, const DistSpCSC<IT, NT, OT>&, const CommGrid3D&); \
  template DistSpCOO<IT, NT, OT> Mpi3DSummaSpGEMM<PlusTimesRing<NT>, IT, NT, OT>(     \
      const DistSpDCSC<IT, NT, OT>&, const DistSpDCSC<IT, NT, OT>&, const CommGrid3D&);

#define SPCRAFT_DEFINE_SUMMA_3D(r, product)                                          \
  SPCRAFT_DEFINE_SUMMA_3D_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                BOOST_PP_SEQ_ELEM(1, product),                       \
                                BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Pair index/offset widths exactly as in the local hash kernel.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SUMMA_3D,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_SUMMA_3D
#undef SPCRAFT_DEFINE_SUMMA_3D_TYPES

}  // namespace spcraft
