#pragma once

#include "solver/Lu.h"

namespace spcraft::solver::symbolic
{

enum class ColumnOrdering { Natural, MinimumDegreeAtA, MinimumDegreeAtPlusA, User };

// Owning canonical CSC structure; numerical values are deliberately absent.
template <class IT, class OT = IT>
struct Pattern {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  IT n = 0;
  std::vector<OT> col_ptr{OT{0}};
  std::vector<IT> row_id;
};

template <class IT>
struct Options {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  ColumnOrdering ordering = ColumnOrdering::Natural;
  std::vector<IT> user_order;  //!< new column -> original column; used only for User.
  IT max_supernode_width = 32;
  IT relaxed_columns = 0;  //!< Zero disables relaxed amalgamation.
};

template <class IT, class OT = IT>
struct Analysis {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  Pattern<IT, OT> input_pattern;  //!< Exact original structure for refactorization checks.
  std::vector<IT> col_order;      //!< new column -> original column; explicit identity when natural.
  std::vector<IT> inverse_col_order;
  std::vector<IT> etree;      //!< Parent in the graph of (A*Pc)^T*(A*Pc); -1 denotes a root.
  std::vector<IT> postorder;  //!< DFS ordering of etree vertices; does not change col_order implicitly.
};

template <class IT, class NT, class OT>
Pattern<IT, OT> ExtractPattern(const CscMatrix<IT, NT, OT>& A);

// First implement Natural/User; minimum-degree variants are later graph exercises.
template <class IT, class OT>
std::vector<IT> ComputeColumnOrder(const Pattern<IT, OT>& pattern, const Options<IT>& options = {});

template <class IT, class OT>
std::vector<IT> ComputeEliminationTree(const Pattern<IT, OT>& pattern, const std::vector<IT>& order);

template <class IT>
std::vector<IT> Postorder(const std::vector<IT>& etree);

template <class IT, class OT>
Analysis<IT, OT> Analyze(const Pattern<IT, OT>& pattern, const Options<IT>& options = {});

// Adapter to the existing no-pivot core: here the same order permutes BOTH axes.
template <class IT, class OT>
LuSymbolic<IT, OT> AnalyzeNoPivotPattern(const Pattern<IT, OT>& pattern, const std::vector<IT>& order);

// Produces a closed block-cross plan for the EXISTING no-pivot supernodal/mt APIs.
// It does not predict the row permutation of a future pivoted factorization.
template <class IT, class OT>
SupernodalSymbolic<IT, OT> DetectSupernodes(const LuSymbolic<IT, OT>& plan, const Options<IT>& options = {});

}  // namespace spcraft::solver::symbolic
