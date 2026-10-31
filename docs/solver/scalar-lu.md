# Algorithm: sparse left-looking LU

Sparse LU solves $Ax=b$ by factoring the matrix into lower and upper triangular
factors while storing only their planned entries. A left-looking algorithm
loads one column into a sparse accumulator, applies the completed columns to its
left, and stores the resulting factor column. Elimination can create **fill**:
nonzero factor entries at positions absent from the input matrix.

## Data structures

The [storage recap](foundations.md#c-storage-preparation) introduces the ownership
rules. Read the declarations linked below alongside the operations used here.

### CscMatrix: input and factor columns

[`CscMatrix<IT,NT,OT>`](../../include/core/CscMatrix.h) holds A, L, and U.
Column k occupies positions `col_ptr[k]` through `col_ptr[k+1]-1`; each position p
contains coordinate `row_id[p]` and coefficient `val[p]`. IT holds coordinates;
OT holds positions and stored-entry counts. This is different from a dense
matrix offset: a matrix coordinate does not directly give a storage position.

`Allocate(nnz,rows,cols)` prepares owning storage. Copy the supplied column
pointers and row indices into factor buffers before numerical work. Keep A
unchanged. L and U use separate CSC matrices; L stores its unit diagonal
explicitly, unlike the implicit diagonal in the packed dense LU buffer.

### LuSymbolic: ordering and space for fill

[`LuSymbolic`](../../include/solver/Lu.h) supplies the L/U patterns, including
fill, and an optional ordering `order[new]=old`. The first exercise uses identity
ordering. With an order, factor $B=P^TAP$, so
$B_{ij}=A_{\mathrm{order}[i],\mathrm{order}[j]}$. An empty order means identity.

The driver copies these patterns into the factor matrices. Numerical kernels
keep every planned slot, including entries that cancel to zero. The sorted rows
in the planned U column tell `UpdateColumn` which earlier pivots to visit.

### Workspace: the sparse accumulator

[`Workspace<IT,NT>`](../../include/solver/scalar/Lu.h) owns four arrays:

- `values[i]`: the current column's residual at ordered row i.
- `marked[i]`: whether row i has been recorded during this column.
- `touched`: the rows to reset before the next column.
- `inverse_order[old]`: the ordered row for an original input row.

`Resize(n)` initializes storage once for a factorization and sets the inverse map
to identity. The driver then fills a nonidentity map when needed.
`ClearColumn()` zeros the touched values and markers while preserving that map
and the allocated buffers. Do not call `Resize` for every column: it would erase
the permutation.

This workspace uses a dense vector of length n for convenient indexing. The
matrix and factors remain sparse, and traversal/cleanup visit stored or touched
entries rather than a dense n-by-n matrix.

### Factors and numerical status

[`Factors`](../../include/solver/scalar/Lu.h) owns L, U, the order, and the overall
status. `StoreColumn` returns a column's `LuStatus`; the driver decides whether
the entire factorization succeeded. A rejected pivot reports `ZeroPivot` and its
ordered column before division. `Solve` reuses successful factors for new RHSs.

## Toy example: fill in a 4-by-4 matrix

Use identity ordering and zero pivot tolerance. The inputs and expected solution
are:

```text
A = [4 1 1 1]     b = [13]     x = [1]
    [1 4 0 0]         [ 9]         [2]
    [1 0 4 0]         [13]         [3]
    [1 0 0 4]         [17]         [4]

A.col_ptr = {0,4,6,8,10}
A.row_id  = {0,1,2,3, 0,1, 0,2, 0,3}
A.val     = {4,1,1,1, 1,4, 1,4, 1,4}
```

Only ten input entries are stored. The displayed zeros are absent CSC entries.
The expected factors are:

```text
L = [ 1      0      0    0]    U = [4   1      1       1  ]
    [1/4     1      0    0]        [0 15/4   -1/4    -1/4 ]
    [1/4  -1/15     1    0]        [0   0    56/15   -4/15]
    [1/4  -1/15  -1/14   1]        [0   0      0      26/7]
```

### 1. Finish column 0

Scatter A's first column into `values={4,1,1,1}`. There are no earlier columns,
so store `U(0,0)=4`, `L(0,0)=1`, and the three multipliers `1/4` below it. A
multiplier tells how much of this pivot column contributes to a later column.
Then clear the touched workspace rows, retaining the completed factor column.

### 2. Scatter column 1

A's second CSC column contains only rows 0 and 1:

```text
values  = {1,4,0,0}
marked  = {1,1,0,0}
touched = {0,1}
```

The two zeros do not mean no work will reach those rows. The planned factors
reserve positions for fill below the diagonal.

### 3. Look left and apply column 0

The current `values[0]=1` is `U(0,1)`. Its contribution comes from completed L
column 0. Skip L's diagonal and subtract its strict-lower entries:

$$
\begin{aligned}
\mathrm{values}[1]&=4-\tfrac14\cdot1=\tfrac{15}{4},\\
\mathrm{values}[2]&=0-\tfrac14\cdot1=-\tfrac14,\\
\mathrm{values}[3]&=0-\tfrac14\cdot1=-\tfrac14.
\end{aligned}
$$

The accumulator is now `{1,15/4,-1/4,-1/4}`. Rows 2 and 3 are newly touched,
so mark and record them. Their nonzero values are fill: A had no stored entries
there. Marking is structural bookkeeping, not a test of whether a value is zero.

### 4. Store column 1, then clean scratch

The updated pivot is `15/4`. After checking it, store:

```text
U(:,1): rows {0,1},   values {1,15/4}
L(:,1): rows {1,2,3}, values {1,-1/15,-1/15}
```

For example, $L_{21}=(-1/4)/(15/4)=-1/15$. `StoreColumn` writes these preallocated
slots. `FactorColumn` then calls `ClearColumn`, including on a pivot failure,
so the next column starts with clean numerical scratch and the same inverse map.

### 5. Repeat with all earlier contributors

For column 2, scattering gives `{1,0,4,0}`. Applying L column 0 changes it to
`{1,-1/4,15/4,-1/4}`. Only now is `U(1,2)=-1/4` ready. Apply L column 1 next:

$$
\mathrm{values}[2]=\tfrac{15}{4}-\left(-\tfrac1{15}\right)\left(-\tfrac14\right)
=\tfrac{56}{15},\qquad
\mathrm{values}[3]=-\tfrac14-\left(-\tfrac1{15}\right)\left(-\tfrac14\right)
=-\tfrac4{15}.
$$

Thus $L_{32}=(-4/15)/(56/15)=-1/14$. Continuing through column 3 gives the
factors shown above. Reading a U coefficient before its earlier contributors
have updated it would use the wrong residual.

Check $LU=A$. Forward substitution gives
$y=\{13,23/4,152/15,104/7\}$, and backward substitution gives
$x=\{1,2,3,4\}$. Finally check the original-system residual $b-Ax$.

## Implement in this order

Work in [`Lu.cpp`](../../src/solver/scalar/Lu.cpp), following
[`Lu.h`](../../include/solver/scalar/Lu.h). The numerical bodies are exercises;
comment out a placeholder exception after completing its function.

| Function | Work to implement | Independent checkpoint |
|---|---|---|
| [`ForwardSolve`](../../src/solver/scalar/Lu.cpp) | Copy rhs; propagate each solved entry down a unit-lower CSC column | Known L and y; no factorization needed |
| [`BackwardSolve`](../../src/solver/scalar/Lu.cpp) | Visit U columns backward; divide and update earlier rows | Known U; include zero/nonfinite pivots |
| [`ScatterColumn`](../../src/solver/scalar/Lu.cpp) | Load an original column into ordered workspace rows | Nonidentity inverse map and unchanged input |
| [`UpdateColumn`](../../src/solver/scalar/Lu.cpp) | Apply earlier L columns in contributor order | The fill trace above; unfinished factor values are poisoned with NaNs |
| [`StoreColumn`](../../src/solver/scalar/Lu.cpp) | Check pivot and write U plus unit-lower L into planned slots | Hand-supplied accumulator; no scatter/update calls |
| [`FactorColumn`](../../src/solver/scalar/Lu.cpp) | Connect scatter, update, store, and cleanup | Full columns, fill, cancellation, and permutation reuse |
| [`Factorize`](../../src/solver/scalar/Lu.cpp) | Allocate patterns, initialize the inverse map once, and run columns | Reconstruct factors and report rejected pivots |
| [`Solve`](../../src/solver/scalar/Lu.cpp) | Gather ordered b, perform both solves, scatter original x | Residual and factor reuse for a second RHS |

`Resize` and `ClearColumn` are implemented infrastructure. Internal column phases
assume valid buffers, patterns, and initialized workspace; avoid repeating shape
and storage scans inside them. The public driver retains the existing input
contract in its declaration. Keep numerical pivot checks before division.

## SuperLU MT source map

The reviewed reference is SuperLU_MT commit
`f2e50bfcbb7b20ef304c84cf92c038c283019863`, available locally at
`refernce/sparse-direct/superlu_mt`. Read these pinned upstream routines after
tracing the corresponding column operation:

| This scaffold | SuperLU_MT reference | Additional concepts to study |
|---|---|---|
| Scatter into `values` | [`pdgstrf_panel_dfs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_panel_dfs.c) | Panel-wide SPA and structural discovery |
| Supplied contributor order | [`pdgstrf_column_dfs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_column_dfs.c) | DFS, segments, markers, and supernode detection |
| `UpdateColumn` | [`pdgstrf_column_bmod.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_column_bmod.c) | Scalar and dense supernode-to-column updates |
| `StoreColumn` | [`pdgstrf_pivotL.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_pivotL.c), [`pdgstrf_copy_to_ucol.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_copy_to_ucol.c) | Threshold row selection, multiplier division, and U-segment storage |
| Column orchestration | [`pdgstrf_thread.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_thread.c) | Panels, private work arrays, and concurrent factorization |
| Reusable triangular solves | [`dgstrs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/dgstrs.c) | Supernodal factor storage and separate row/column permutations |

This first implementation uses plain CSC factors, a supplied fill pattern,
symmetric ordering, and no row pivoting. SuperLU_MT performs supernodal/panel
factorization with structural discovery and threshold partial row pivoting,
using separate row and column permutations. Its threshold is relative to a
column maximum; this exercise's `pivot_tolerance` is an absolute rejection rule.
The source map connects the computations without equating the storage contracts.

After the scalar kernels work, continue with [supernodal LU](supernodal-lu.md),
then [parallel LU](parallel-lu.md). Keep reach/DFS discovery and threshold row
pivoting as subsequent exercises, using the existing [source-reading lessons](source-reading.md).

## Check and explain

Build `test_solver_scalar`; run the phase checkpoints in
[the checkpoint guide](checkpoints.md#scalar-checkpoints). Expected numerical
failures remain visible until the corresponding exercise is implemented.

Before moving on, explain:

1. Why does A's column 1 have two stored entries while L's has three?
2. Why must `values[1]` be updated before reading it as `U(1,2)`?
3. Why are `Resize` and `ClearColumn` different operations?
4. Why can a planned factor slot contain a numerical zero?
5. Where would SuperLU_MT's row-pivot maps enter this simpler coordinate model?
