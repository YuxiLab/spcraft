#include "mpi/DistSpCSC.h"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "mpi/DistSpCommon.h"

namespace spcraft
{

template <class IT, class NT, class OT>
DISTSPCSC::DistSpCSC() noexcept = default;

template <class IT, class NT, class OT>
DISTSPCSC::DistSpCSC(GridPtr grid, IT global_m, IT global_n) : commgrid(std::move(grid))
{
  Allocate(OT{0}, global_m, global_n);
}

template <class IT, class NT, class OT>
DISTSPCSC::DistSpCSC(GridPtr grid, MatrixType&& data, IT global_m, IT global_n) : commgrid(std::move(grid))
{
  const std::pair<IT, IT> dims = detail::DistLocalDimensions(commgrid, global_m, global_n);
  if (data.m != dims.first || data.n != dims.second) {
    throw std::invalid_argument("Local matrix dimensions do not match the process-grid partition");
  }
  m = global_m;
  n = global_n;
  local = std::move(data);
}

template <class IT, class NT, class OT>
DISTSPCSC::DistSpCSC(DISTSPCSC&& rhs) noexcept { *this = std::move(rhs); }

template <class IT, class NT, class OT>
DISTSPCSC& DISTSPCSC::operator=(DISTSPCSC&& rhs) noexcept
{
  if (this != &rhs) {
    local = std::move(rhs.local);
    commgrid = std::move(rhs.commgrid);
    m = rhs.m;
    n = rhs.n;
    rhs.Reset();
  }
  return *this;
}

template <class IT, class NT, class OT>
DISTSPCSC::~DistSpCSC() = default;

template <class IT, class NT, class OT>
void DISTSPCSC::Allocate(OT count) { Allocate(count, m, n); }

template <class IT, class NT, class OT>
void DISTSPCSC::Allocate(OT count, IT global_m, IT global_n)
{
  const std::pair<IT, IT> dims = detail::DistLocalDimensions(commgrid, global_m, global_n);
  CscMatrix<IT, NT, OT> replacement;
  replacement.Allocate(count, dims.first, dims.second);
  local = std::move(replacement);
  m = global_m;
  n = global_n;
}

template <class IT, class NT, class OT>
DISTSPCSC DISTSPCSC::Clone() const
{
  DISTSPCSC result;
  if (!commgrid) return result;
  CscMatrix<IT, NT, OT> copied = local.Clone();
  result = DISTSPCSC(commgrid, std::move(copied), m, n);
  return result;
}

template <class IT, class NT, class OT>
void DISTSPCSC::Reset() noexcept
{
  local.Reset();
  commgrid.reset();
  m = 0;
  n = 0;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::LocalRows() const
{
  if (!commgrid) return IT{0};
  const IT rows = detail::LocalExtent(m, commgrid->Rows(), commgrid->Row());
  return rows;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::LocalCols() const
{
  if (!commgrid) return IT{0};
  const IT cols = detail::LocalExtent(n, commgrid->Cols(), commgrid->Col());
  return cols;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::GlobalRow(IT local_row) const
{
  if constexpr (std::is_signed_v<IT>) {
    if (local_row < IT{0}) throw std::out_of_range("Local row is outside the matrix");
  }
  if (local_row >= LocalRows()) throw std::out_of_range("Local row is outside the matrix");
  const IT row = detail::GlobalIndex(local_row, m, commgrid->Rows(), commgrid->Row());
  return row;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::GlobalCol(IT local_col) const
{
  if constexpr (std::is_signed_v<IT>) {
    if (local_col < IT{0}) throw std::out_of_range("Local column is outside the matrix");
  }
  if (local_col >= LocalCols()) throw std::out_of_range("Local column is outside the matrix");
  const IT col = detail::GlobalIndex(local_col, n, commgrid->Cols(), commgrid->Col());
  return col;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::LocalRow(IT global_row) const
{
  if (!commgrid) throw std::logic_error("Distributed matrix has no grid");
  const IT row = detail::LocalIndex(global_row, m, commgrid->Rows(), commgrid->Row());
  return row;
}

template <class IT, class NT, class OT>
IT DISTSPCSC::LocalCol(IT global_col) const
{
  if (!commgrid) throw std::logic_error("Distributed matrix has no grid");
  const IT col = detail::LocalIndex(global_col, n, commgrid->Cols(), commgrid->Col());
  return col;
}

template <class IT, class NT, class OT>
int DISTSPCSC::Owner(IT global_row, IT global_col) const
{
  if (!commgrid) throw std::logic_error("Distributed matrix has no grid");
  if constexpr (std::is_signed_v<IT>) {
    if (global_row < IT{0} || global_col < IT{0}) {
      throw std::out_of_range("Global entry is outside the matrix");
    }
  }
  if (global_row >= m || global_col >= n) {
    throw std::out_of_range("Global entry is outside the matrix");
  }
  const IT row_extent = m / commgrid->Rows() + (m % commgrid->Rows() != 0);
  const IT col_extent = n / commgrid->Cols() + (n % commgrid->Cols() != 0);
  const int row = global_row / row_extent;
  const int col = global_col / col_extent;
  return commgrid->RankAt(row, col);
}

template <class IT, class NT, class OT>
bool DISTSPCSC::Owns(IT global_row, IT global_col) const
{
  const int owner = Owner(global_row, global_col);
  return owner == commgrid->Rank();
}

template <class IT, class NT, class OT>
int64_t DISTSPCSC::GlobalNnz() const
{
  if (!commgrid) throw std::logic_error("Distributed matrix has no grid");
  const int64_t global_nnz = detail::GlobalNnz(*commgrid, local.nnz);
  return global_nnz;
}

#define SPCRAFT_DEFINE_DIST_CSC(r, product)                                     \
  template class DistSpCSC<BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                           BOOST_PP_SEQ_ELEM(1, product),                       \
                           BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product))>;

// Match the local containers, including Boolean graph storage.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_DIST_CSC,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)(bool)))
// clang-format on
#undef SPCRAFT_DEFINE_DIST_CSC

}  // namespace spcraft
