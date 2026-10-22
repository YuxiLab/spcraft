#pragma once

#include <vector>

#include "core/CscMatrix.h"
#include "core/DenseVector.h"

namespace spcraft::solver
{

enum class LuCode { NotFactorized, Success, ZeroPivot };

template <class IT>
struct LuStatus {
  LuCode code = LuCode::NotFactorized;
  IT pivot_column = IT{-1};  //!< Ordered column on failure; -1 otherwise.
};

template <class NT>
struct LuOptions {
  NT pivot_tolerance = NT{0};  //!< Reject |pivot| <= tolerance; never perturb a pivot.
};

/**
 * @brief Instructor-supplied structural plan for no-pivot LU of B = P^T A P.
 *
 * order[new] = old, so B(i,j) = A(order[i],order[j]). Empty order means identity.
 * Patterns are zero-based CSC, sorted and duplicate-free, include both diagonals,
 * and contain every possible fill entry. Keep numerical cancellations as stored
 * zeros. L has an explicitly stored unit diagonal. No row pivoting is supported.
 * The empty problem has n=0, empty row arrays, and pointer arrays containing {0}.
 */
template <class IT, class OT = IT>
struct LuSymbolic {
  IT n = 0;
  std::vector<IT> order;
  std::vector<OT> lower_col_ptr;
  std::vector<IT> lower_row_id;
  std::vector<OT> upper_col_ptr;
  std::vector<IT> upper_row_id;
};

/**
 * @brief One supernode's structural cross in the ordered matrix.
 *
 * Columns [first,last) form the diagonal block. rows lists indices >= last
 * present below it; columns lists indices >= last present to its right.
 * Both lists are sorted and unique. contributors contains ALL earlier panels
 * whose completed factors are read when updating this cross, once each, sorted
 * by panel ID. It is a numerical read-dependency list, not merely etree children.
 */
template <class IT>
struct SupernodePattern {
  IT first = 0;
  IT last = 0;
  std::vector<IT> rows;
  std::vector<IT> columns;
  std::vector<IT> contributors;
};

/**
 * @brief Instructor-supplied block plan; panels partition [0, scalar.n).
 *
 * Dense panel blocks may include structural zeros. The cross patterns must be
 * closed under all contributor updates. Panel IDs follow topological order.
 * The plan is immutable throughout numerical factorization and safe to share.
 */
template <class IT, class OT = IT>
struct SupernodalSymbolic {
  LuSymbolic<IT, OT> scalar;
  std::vector<SupernodePattern<IT>> panels;
};

}  // namespace spcraft::solver
