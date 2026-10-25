# LU foundations

Use this chapter before the C++ exercises. You need matrix multiplication,
solving a small linear system, and basic loops/arrays. Before the sparse phase,
learn CSC offsets and sorted row IDs. Before the parallel phase, learn mutexes,
data races, and the difference between a task dependency and a worker thread.

## 1. Elimination stores a reusable transformation

We want to solve $Ax=b$. Gaussian elimination removes entries below each pivot.
At step $k$, the multiplier for row $i$ is $\ell_{ik}=a_{ik}/a_{kk}$, using the values
after previous elimination steps. Subtracting that multiple of pivot row k
changes the remaining entries by

$$
a_{ij}\leftarrow a_{ij}-\ell_{ik}u_{kj},\qquad i>k,\quad j>k.
$$

U stores the remaining upper triangle; L stores the multipliers with a unit
diagonal. With no row swaps, $A=LU$. Factoring A once allows many RHSs to use
the same L and U. A solve must not repeat elimination on A for every RHS.

For a leading block, multiply the factors to derive

$$
\begin{bmatrix} A_{11}&A_{12}\\ A_{21}&A_{22}\end{bmatrix}
=
\begin{bmatrix} L_{11}&0\\ L_{21}&L_{22}\end{bmatrix}
\begin{bmatrix} U_{11}&U_{12}\\ 0&U_{22}\end{bmatrix}.
$$

$$
\begin{aligned}
A_{11}&=L_{11}U_{11}, & A_{21}&=L_{21}U_{11},\\
A_{12}&=L_{11}U_{12}, & S&=A_{22}-L_{21}U_{12}=L_{22}U_{22}.
\end{aligned}
$$

S is the Schur complement left for later elimination. A scalar pivot is the
width-one version of this block calculation. Derive these four equations before
studying TRSM or GEMM names.

## 2. Triangular solves explain elimination order

First solve $Ly=b$, then $Ux=y$. A unit-lower solve never divides by the stored
diagonal. In the CSC column algorithm, when column j is reached, earlier columns
have already subtracted their contributions from y[j]. Propagate the now-known
y[j] only to rows below it.

For U, visit columns in reverse order. Divide the residual at j by U(j,j), then
subtract its contribution from earlier rows. A reverse loop must also handle
n=0. If a diagonal is missing, zero, or nonfinite, do not divide.

Exercise: explain why subtracting L(j,j)*y[j] would destroy a correct unit-lower
solution. Then trace the two solves in [the worked example](worked-example.md).

## 3. Sparsity changes storage and the work you visit

CSC stores column j in positions `[col_ptr[j],col_ptr[j+1])`. Each position p
holds row `row_id[p]` and coefficient `val[p]`. A storage position and a matrix
coordinate are different quantities. Here IT holds coordinates and OT holds
offsets/counts, allowing a matrix's number of stored entries to exceed a 32-bit
offset even when its dimensions do not.

Elimination can create a nonzero where A has none: fill. The symbolic phase
predicts structural positions; the numerical phase computes values there.
Cancellation can make a predicted entry zero. Its planned slot remains present.
Explicit numerical zero, absent input entry, and planned factor slot are three
different ideas.

A sparse accumulator uses a dense values array for convenient row access plus
markers and a touched list. It is sparse in the work visited and reset. Before
extracting the next U(k,j), the current accumulator must include every preceding
contributor update. L's completed columns then supply the numerical subtraction.

Exercise: locate the first fill entry in the 3-by-3 example and explain why
copying only A's original sparsity into the factor buffers cannot work.

### C++ storage preparation

`CscMatrix::Allocate(nnz, nRows, nCols)` allocates storage; the first argument
is the entry count. Fill its pointer, row, and value arrays before using them.
`DenseVector(n)` allocates uninitialized values, while `DenseVector(n, value)`
initializes them. Matrices use `DenseMatrix(rows,cols)` or the initialized
`DenseMatrix(rows,cols,value)`. Access entries with `A[row,col]` in C++23;
`A(row,col)` is also available. Column-major
offsets are `row+col*A.ld`. Packed owners have `ld==rows`, while a view may have
padding between columns. Check these contracts before assuming scratch is zero.

The matrix/vector owners used here support moves rather than ordinary copies.
Read `core/CscMatrix.h`, `core/DenseMatrix.h`, and `core/DenseVector.h` to distinguish an owning buffer
from a view. Moving a completed panel transfers ownership; it does not make
concurrent reads safe by itself. Numerical ownership and thread synchronization
are separate responsibilities.

## 4. Keep three coordinate systems distinct

The assignment uses a symmetric permutation, even for nonsymmetric matrices:

$$
\begin{aligned}
\texttt{order[new]}&=\texttt{old},\\
\texttt{inverse\_order[old]}&=\texttt{new},\\
B_{ij}&=A_{\texttt{order[i]},\texttt{order[j]}}=(P^TAP)_{ij},\\
\texttt{rhs[i]}&=\texttt{b[order[i]]},\\
LUz&=\mathrm{rhs},\\
\texttt{x[order[i]]}&=\texttt{z[i]}.
\end{aligned}
$$

Original coordinates index A and b. Ordered coordinates index L/U and the plan.
Local coordinates index a dense panel buffer. A global row r may have local
position 0 in one panel and position 2 in another. Always use its map before
indexing a source buffer.

Exercise: for `order={2,0,1}`, compute `inverse_order`, ordered b, and the final
scatter without introducing another permutation convention. Production solvers
can have separate row and column permutations; study those after the core solve.

## 5. Block structure makes dense kernels possible

A supernode groups consecutive columns whose factor structure can share dense
storage below the diagonal. A fundamental supernode follows structural matching;
relaxed grouping may include extra zeros to create useful larger blocks.

In this scaffold each `Panel` is one planned column interval with a diagonal
block, lower-row map, and upper-column map. It is both a storage and scheduling
unit. Its representation is deliberately simple. SuperLU_MT also uses numerical
panels that need not coincide with individual structural supernodes.

`FactorPanel` performs diagonal LU and two block triangular solves. `UpdatePanel`
subtracts products from earlier panels. TRSM names a triangular matrix solve;
GEMM names a dense matrix product with scaling/accumulation. Larger kernels may
reuse data more effectively, but gathering, padding, and small blocks can erase
that benefit. Measure rather than assuming block code is faster.

Exercise: derive the dimensions of all three products in the block worked
example. Explain why `rows` and `columns` need not be equal.

## 6. Parallelism comes from mathematical dependencies

Independent panels may execute concurrently. A destination that reads two source
panels waits for both to finish and publish their values. An elimination tree
encodes structural ancestry; the scaffold explicitly supplies every numerical
read dependency in `contributors`.

Keep a task's private workspace and output separate from the scheduler's queue
and counters. One writer owns each output panel. Completion synchronization must
make buffer writes visible before a successor reads them. A mutex can provide
that ordering if producers and consumers follow the same protocol. Merely
setting a flag, or declaring it volatile, does not establish the full protocol.

Exercise: draw $0\rightarrow2\leftarrow1$. Describe two workers finishing in either order,
an empty ready queue with one worker still active, and a failed predecessor.

## 7. Stability and error are part of correctness

The core exercises use SPD or strictly diagonally dominant inputs and reject
nonfinite pivots or `abs(pivot)<=tolerance`. Finite precision still matters.
`[[0,1],[1,0]]` is nonsingular but cannot start this no-pivot elimination.

SuperLU_MT's threshold row pivoting chooses a suitable row relative to a column
maximum. PaStiX's static-pivoting kernels can replace a small diagonal. The
assignment's absolute rejection tolerance is a different policy. Follow the
[pivoting lesson](source-reading.md#lesson-g-pivoting-and-numerical-stability)
before changing any policy.

For a computed solution, form $r=b-Ax$ with the original A and b. Report

$$
\begin{aligned}
\text{forward error}
  &=\frac{\lVert x-x_{\mathrm{true}}\rVert_\infty}{\lVert x_{\mathrm{true}}\rVert_\infty},\\[4pt]
\text{backward error}
  &=\frac{\lVert r\rVert_\infty}{\lVert A\rVert_\infty\lVert x\rVert_\infty+\lVert b\rVert_\infty},\\[4pt]
\lVert A\rVert_\infty&=\max_i\sum_j|a_{ij}|.
\end{aligned}
$$

The matrix infinity norm is the maximum absolute row sum.

Handle a zero denominator: report zero if the numerator is zero, otherwise
infinity. A small backward error measures a small scaled residual; a poorly
conditioned system can still have substantial forward error. Factor
reconstruction checks the factorization itself; the original residual also
checks permutation and solve logic. Both are useful.

Return to the [implementation sequence](README.md#stage-0-understand-lu-before-implementing-sparse-lu)
when you can explain these distinctions in your own words.
