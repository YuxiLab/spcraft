#pragma once

#include "solver/symbolic/Lu.h"

namespace spcraft::solver::pivoted
{

enum class Operation { Normal, Transpose, ConjugateTranspose };

template <class IT, class NT>
struct Options {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  NT diagonal_threshold = NT{1};  //!< Relative to the largest active column entry; in [0,1].
  NT pivot_tolerance = NT{0};     //!< Absolute rejection threshold; never replaces a pivot.
  std::vector<IT> row_hint;       //!< Optional proposed pivot row per column, new -> original.
};

template <class IT, class NT>
struct Column {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::vector<IT> rows;
  std::vector<NT> values;
};

// Building L uses ORIGINAL row IDs, so later row swaps never rewrite old columns.
// Building U uses pivot-step row IDs. Only FinalizeFactors converts L to pivot coordinates.
template <class IT, class NT, class OT = IT>
struct Assembly {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT n = 0;
  std::vector<IT> col_order;
  std::vector<IT> row_order;          //!< pivot step -> original row; unassigned slots are -1.
  std::vector<IT> inverse_row_order;  //!< original row -> pivot step; unpivoted rows are -1.
  std::vector<Column<IT, NT>> lower;
  std::vector<Column<IT, NT>> upper;
  LuStatus<IT> status;
};

template <class IT, class NT>
struct Workspace {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::vector<NT> values;  //!< Accumulator indexed by ORIGINAL row.
  std::vector<unsigned char> marked;
  std::vector<IT> touched;
  std::vector<IT> reach;  //!< Completed pivot columns in increasing elimination order.
};

template <class IT, class NT>
struct Pivot {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT old_row = IT{-1};
  NT value = NT{0};
  LuStatus<IT> status;
};

// Finalized canonical CSC: A(row_order[i],col_order[j]) = (L*U)(i,j).
// L has an explicit unit diagonal. Both permutations are new -> original.
template <class IT, class NT, class OT = IT>
struct Factors {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  CscMatrix<IT, NT, OT> lower;
  CscMatrix<IT, NT, OT> upper;
  std::vector<IT> row_order;
  std::vector<IT> col_order;
  LuStatus<IT> status;
};

template <class IT, class NT, class OT>
Assembly<IT, NT, OT> Initialize(const symbolic::Analysis<IT, OT>& analysis);

// Reserve paired row/value capacity, preserving all entries and logical sizes.
template <class IT, class NT, class OT>
void ReserveColumn(Column<IT, NT>& column, OT required);

template <class IT, class NT, class OT>
void ScatterColumn(const CscMatrix<IT, NT, OT>& A, IT old_col, Workspace<IT, NT>& work);

// Discover dependencies from input structure and completed L, without numerical zero tests.
template <class IT, class NT, class OT>
std::vector<IT> ReachColumn(const CscMatrix<IT, NT, OT>& A, IT j, const Assembly<IT, NT, OT>& assembly);

template <class IT, class NT, class OT>
void UpdateColumn(IT j, const Assembly<IT, NT, OT>& assembly, Workspace<IT, NT>& work);

template <class IT, class NT, class OT>
Pivot<IT, NT> SelectPivot(IT j, const Assembly<IT, NT, OT>& assembly, const Workspace<IT, NT>& work,
                          const Options<IT, NT>& options = {});

template <class IT, class NT, class OT>
void StoreColumn(IT j, const Pivot<IT, NT>& pivot, Assembly<IT, NT, OT>& assembly,
                 const Workspace<IT, NT>& work);

// Consumes the building storage; output is immutable to Solve and uses final pivot coordinates.
template <class IT, class NT, class OT>
Factors<IT, NT, OT> FinalizeFactors(Assembly<IT, NT, OT>&& assembly);

template <class IT, class NT, class OT>
Factors<IT, NT, OT> Factorize(const CscMatrix<IT, NT, OT>& A, const symbolic::Analysis<IT, OT>& analysis,
                              const Options<IT, NT>& options = {});

// B/X are disjoint n-by-nrhs dense matrices; preserve factors and B, including padded leading dimensions.
template <class IT, class NT, class OT>
void Solve(const Factors<IT, NT, OT>& factors, const DenseMatrix<IT, NT, OT>& B, DenseMatrix<IT, NT, OT>& X,
           Operation operation = Operation::Normal);

}  // namespace spcraft::solver::pivoted
