# Algorithm: sparse LU analysis, pivoting, and accurate solves

A general sparse LU solver chooses a column order to reduce fill, selects row
pivots during elimination, and reuses its factors to solve linear systems.
Scaling and iterative refinement help with numerical accuracy. These exercises
extend the existing fixed-pattern core toward SuperLU_MT's concepts. Finish the
current scalar, supernodal, and parallel kernels first; all new algorithm bodies
are exercises that throw until implemented.

## Data structures

- [`symbolic::Pattern` and `Analysis`](../../include/solver/symbolic/Lu.h)
  own the canonical CSC structure, column ordering, elimination tree, and tree
  postorder. `ExtractPattern` copies positions without values; `Analyze` prepares
  metadata reusable when values change. As in
  [the CSC storage recap](scalar-lu.md#data-structures), column pointers locate
  each column and row IDs name its stored positions. Explicit numerical zeros
  remain structural entries.
- [`pivoted::Assembly`](../../include/solver/pivoted/Lu.h) owns growing L/U
  columns. `ReserveColumn` grows paired row/value buffers without changing their
  logical sizes. Building L uses original row IDs; building U uses completed
  pivot indices. `row_order[k]` names the original row chosen at pivot k;
  `inverse_row_order[old_row]` is -1 until that row is chosen.
- [`pivoted::Workspace`](../../include/solver/pivoted/Lu.h) contains an
  accumulator indexed by original rows, touched-row markers, and a dependency
  reach list. `ScatterColumn` loads input; `ReachColumn` discovers contributing
  completed columns; `UpdateColumn` applies them. The fixed-pattern
  [scalar workspace](../../include/solver/scalar/Lu.h) remains a separate,
  simpler exercise.
- [`pivoted::Factors`](../../include/solver/pivoted/Lu.h) owns finalized CSC
  L/U and independent row/column permutations. `FinalizeFactors` maps building
  L rows into final pivot coordinates and sorts them. `Solve` takes column-major
  [dense matrices](../../include/core/DenseMatrix.h) so several RHSs can share
  the factors; use `B[i,j]`, `X[i,j]`, and each matrix's own leading dimension.
- [`expert::State`](../../include/solver/expert/Lu.h) groups reusable analysis,
  positive row/column scales, and factors of the scaled matrix. `Diagnostics`
  reports accuracy per RHS; refinement status distinguishes convergence,
  stagnation, iteration limits, and nonfinite arithmetic.

## A four-by-four example with row pivots and fill

Take

$$
A=\begin{bmatrix}
0&2&1&0\\
1&3&0&0\\
0&0&0&4\\
0&0&2&5
\end{bmatrix},\qquad
x=\begin{bmatrix}1\\2\\3\\4\end{bmatrix},\qquad
b=Ax=\begin{bmatrix}7\\7\\16\\26\end{bmatrix}.
$$

Supply `col_order={2,3,0,1}`, meaning new column j comes from original column
`col_order[j]`. Column ordering alone gives

$$
AP_c=\begin{bmatrix}
1&0&0&2\\
0&0&1&3\\
0&4&0&0\\
2&5&0&0
\end{bmatrix}.
$$

Use `diagonal_threshold=1` and `pivot_tolerance=0`. The former compares a
preferred pivot with the largest active column magnitude; the latter rejects a
numerically unusable pivot. They serve different purposes.

1. **Column 0, original column 2.** Original row 2 has a zero diagonal
   candidate. Original row 3 has the largest magnitude, 2, so choose row 3.
   Original row 0 contains 1. Its multiplier is the amount of the pivot row
   needed to eliminate that entry: solve `1 - multiplier*2 = 0`, obtaining 1/2.
   Subtracting half the pivot row changes original row 0's remaining entries
   from `[0,0,2]` to `[-5/2,0,2]`. The new -5/2 is fill.
2. **Column 1, original column 3.** Scattering gives original-row scratch
   `[0,0,4,5]`. Reach finds pivot 0. Its coefficient is 5, and the previous
   multiplier changes row 0 to `0-(1/2)*5=-5/2`, reproducing the delayed update.
   Original row 3 is already used, so choose row 2 with pivot 4. The remaining
   row 0 multiplier is `(-5/2)/4=-5/8`.
3. **Column 2, original column 0.** Original row 1 supplies pivot 1. The
   remaining row 0 coefficient is zero, so no nonzero multiplier is needed.
4. **Column 3, original column 1.** The entry 3 in original row 1 belongs to
   U at pivot row 2. Original row 0 supplies the last pivot, 2.

Thus `row_order={3,2,1,0}`. With both orders stored as **new to original**,

$$
A(\text{row\_order},\text{col\_order})=LU,
$$

$$
L=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
1/2&-5/8&0&1
\end{bmatrix},\qquad
U=\begin{bmatrix}
2&5&0&0\\
0&4&0&0\\
0&0&1&3\\
0&0&0&2
\end{bmatrix}.
$$

Check the last product row: `(1/2)*U(0,:) - (5/8)*U(1,:) + U(3,:)`
is `[1,0,0,2]`, which is original row 0 in the chosen column order.
The building L entries for original row 0 stay indexed by 0 until finalization;
only then are they mapped to final row 3. This avoids rewriting earlier columns
when later pivots are chosen.

For a normal solve, gather `b[row_order]={26,16,7,7}`. Forward substitution gives
`y={26,16,7,4}`. Backward substitution gives `z={3,4,1,2}`. Scattering
`x[col_order[j]]=z[j]` restores `{1,2,3,4}`. A transpose solve reverses the roles
of the permutations and uses U-transpose before L-transpose. For this same x,
`A^T*x={2,8,9,32}` supplies an independent transpose checkpoint.

## Implementation sequence

The original core interfaces and tests keep their current contracts. These are
additional layers, not changes to a function you are already implementing.

| Finish first | Functions to implement next | Result |
|---|---|---|
| Current scalar LU | [`ExtractPattern`, `ComputeColumnOrder`, `ComputeEliminationTree`, `Postorder`, `Analyze`](../../src/solver/symbolic/Lu.cpp) | Reusable structure, ordering, and forest metadata |
| Structural elimination by hand | [`AnalyzeNoPivotPattern`](../../src/solver/symbolic/Lu.cpp) | Automatically supplied plans for the existing scalar core |
| Current serial supernodal LU | [`DetectSupernodes`](../../src/solver/symbolic/Lu.cpp) | Closed block-cross plans for the existing supernodal and parallel APIs |
| Sparse accumulator and reach | [`Initialize`, `ReserveColumn`, `ScatterColumn`, `ReachColumn`, `UpdateColumn`](../../src/solver/pivoted/Lu.cpp) | Dynamic, original-row factor-building storage |
| Separate permutation mappings | [`SelectPivot`, `StoreColumn`, `FinalizeFactors`, `Factorize`](../../src/solver/pivoted/Lu.cpp) | Serial threshold-pivoted sparse LU |
| Successful supplied pivoted factors | [`pivoted::Solve`](../../src/solver/pivoted/Lu.cpp) | Normal, transpose, and real adjoint solves with several RHSs |
| Pivoted factorization and solve | [`ComputeScaling`, `ApplyScaling`, `Refactorize`, `expert::Solve`](../../src/solver/expert/Lu.cpp) | Scaled numerical state with reusable analysis |
| Scaling-aware normal/adjoint solves | [`EstimateReciprocalCondition`, `ReciprocalPivotGrowth`, `Refine`](../../src/solver/expert/Lu.cpp) | Accuracy and stability diagnostics |

Start ordering with Natural and User. Implement the minimum-degree graph
variants later with structural sets and deterministic tie breaking. They teach
the ordering principle; they do not claim to reproduce SuperLU_MT's optimized
ordering implementations or its exact permutation choices.

`AnalyzeNoPivotPattern` deliberately uses a symmetric permutation to match the
existing `LuSymbolic` contract. General `Analyze` orders columns only. Do not
pass a general column permutation into the old core and assume it represents
the same factorization. The adapters return ordinary `LuSymbolic` and
`SupernodalSymbolic` objects, so existing scalar, supernodal, and mt drivers can
consume automatically discovered **no-pivot** plans without signature changes.

## Scaling, refinement, and reuse

Define `As=R*A*C`, where R and C are positive diagonal scale matrices. Store
their entries explicitly, using ones for identity scales. Factors represent As.

- Normal: solve `As*z=R*b`, then return `x=C*z`.
- Transpose or adjoint: solve `op(As)*z=C*b`, then return `x=R*z`.

For refinement compute the residual against **original** A and b. Solve for a
correction through the scaling-aware expert solve, update x, and recompute the
componentwise backward error. A small residual is not itself a forward error
bound; the forward estimate also requires inverse actions and rounding-error
weights. Refinement can stagnate.

Create a state by assigning the result of `Analyze` to `state.analysis`, then
calling `Refactorize`. That function computes scales and numerical factors.
A second call accepts new values with the exact same canonical CSC structure.
Reusing analysis is different from reusing factors for a new RHS. Passing the
old row order as `row_hint` proposes pivots but still allows threshold tests to
choose different rows. This first refactorization exercise reuses analysis;
retaining allocated factor-buffer capacity is a later optimization.

At public entry points validate structure and options once. Internal kernels
assume valid shapes and maps, while retaining required numerical checks. No
pivot is silently replaced, and `Solve` must reject unsuccessful factors.

## Checkpoints

Enable the existing solver assignments option, then build:

```bash
cmake --build build/solver-learning --target test_solver_extensions
build/solver-learning/bin/test_solver_extensions --gtest_filter='ExtensionFixtures.*'
build/solver-learning/bin/test_solver_extensions --gtest_filter='SymbolicAnalysis.*:SymbolicSupernodes.*'
build/solver-learning/bin/test_solver_extensions --gtest_filter='DynamicFactorStorage.*:ThresholdPivot.*'
build/solver-learning/bin/test_solver_extensions --gtest_filter='PivotedColumn.*:PivotedFactorize.*:PivotedSolve.*'
build/solver-learning/bin/test_solver_extensions --gtest_filter='ExpertSolve.*:ExpertRefinement.*:ExpertRefactorize.*'
```

The independent fixture check passes before any algorithm is implemented.
The other checks are acceptance exercises and currently fail on the active
placeholder exceptions. They check mathematical outputs, not that a stub throws.
Some integrated checks require several earlier exercises; finish direct kernels
first. As with the original core, passing these initial examples alone does not
establish production robustness. Add empty/singular cases, graph-ordering
checks, nonidentity mappings, random residual checks, and all exported integer
widths as the corresponding functions are implemented.

## SuperLU_MT reading map and remaining optimization work

The source reference is pinned to
[`f2e50bfc`](https://github.com/xiaoyeli/superlu_mt/tree/f2e50bfcbb7b20ef304c84cf92c038c283019863).
Read each routine after its matching exercise; this scaffold uses its own C++
storage and contracts rather than wrapping the C library.

| Exercise | SuperLU_MT reference |
|---|---|
| Column ordering and structural preprocessing | [`get_perm_c.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/get_perm_c.c), [`sp_colorder.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/sp_colorder.c) |
| Reach, supernode discovery, and segment ordering | [`pdgstrf_column_dfs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_column_dfs.c) |
| Relative pivot threshold and row mapping | [`pdgstrf_pivotL.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_pivotL.c) |
| Allocation and factor memory growth | [`pdmemory.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdmemory.c) |
| Transpose and multiple RHS solves | [`dgstrs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/dgstrs.c) |
| Scaling, refinement, diagnostics, and refactor options | [`pdgssvx.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgssvx.c) |
| Parent preference and panel pipelining | [`pxgstrf_scheduler.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pxgstrf_scheduler.c) |

The new executable exercises cover analysis, dynamic scalar pivoting, and expert
solves. The existing supernodal/mt exercises still use immutable, no-pivot block
plans. A production pivoted supernodal runtime needs further design after both
paths work: segment-based compressed storage, DFS pruning, dynamic supernode
decisions, synchronized pivot ownership, shared memory growth, and pipelined
publication. Do not combine a mutable row-pivot map with the existing immutable
task graph without defining those dependencies and ownership rules first.

The new source files export `double` with signed `(IT,OT)` widths `(32,32)`,
`(32,64)`, and `(64,64)`. Single precision can follow the same exercises.
Complex exports also require core container instantiations, real-valued
magnitude/scaling/diagnostic types, and actual conjugation in adjoint kernels;
the real `ConjugateTranspose` mode alone does not provide complex support.
This is a growing learning scaffold, not complete SuperLU_MT feature parity.
