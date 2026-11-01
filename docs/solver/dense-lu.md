# Algorithm: dense LU and triangular solves

Dense LU factorization solves a linear system $Ax=b$ by using Gaussian
elimination to write a square matrix as $A=LU$, where L is unit lower triangular
and U is upper triangular. Forward substitution solves $Ly=b$, then backward
substitution solves $Ux=y$. The factors can be reused for new right-hand sides.
This version uses no pivoting, so it reports failure if an updated pivot is
nonfinite or too small under the chosen tolerance.

## Data structures

The [C++ storage recap](foundations.md#c-storage-preparation) introduces the
matrix and vector containers and their ownership conventions. The operations
needed here are summarized below.

### DenseMatrix: input matrix and factor storage

[`DenseMatrix<IT,NT,OT>`](../../include/core/DenseMatrix.h) stores A and the
combined L/U factor buffer. `A.m` and `A.n` give the dimensions; `A[row,col]`
reads or updates a coefficient during elimination and substitution. Storage is
column-major, with offset `row+col*A.ld`. IT holds coordinates, while OT holds
offsets, strides, and the logical entry count returned by `A.size()`.
The project uses C++23 for this two-index bracket syntax. `A(row,col)` also
remains available; vectors use `v[i]`.

- `DenseMatrix(rows,cols)` allocates uninitialized storage; the value-taking
  constructor initializes every entry, which is useful when preparing examples.
- `A.Clone()` creates an independent owning copy for elimination, preserving
  the original A for reuse and residual checks.
- Pointer constructors wrap external storage, optionally with a leading
  dimension. Owners have `ld==m`; views may have `ld>m`. Use `A[row,col]` to
  respect that stride, and visit only logical entries when validating values.

Views borrow live host storage supplied by the caller. Matrix allocation and
view construction check dimensions, stride, and storage-span overflow;
coefficient access itself is unchecked. Ordinary copying is disabled; moves
transfer the matrix state and its ownership responsibility.

### DenseVector: right-hand sides and solution vectors

[`DenseVector<IT,NT>`](../../include/core/DenseVector.h) stores b, the intermediate
y, and the final x. `v.size()` gives the vector length, and `v[i]` accesses an
entry while accumulating a row sum or storing a newly solved unknown.
`DenseVector(n)` allocates uninitialized storage for an output or workspace;
write each entry before reading it. `DenseVector(n,value)` initializes a vector,
and the pointer constructor borrows an existing buffer. Solves require
host-accessible storage and matching lengths. Keep y private and output storage
separate from input buffers, as specified in the caller assumptions below.

### Factors and numerical status

[`dense::Factors<IT,NT,OT>`](../../include/solver/dense/Lu.h) groups an owning
`lu` matrix with a `status`. `RightLookingFactorize(A)` derives dimensions from A and fills
this record; the caller checks its status before passing it to `Solve` for two
substitutions. Each triangular solve likewise derives dimensions from its
factor matrix.

[`LuOptions` and `LuStatus`](../../include/solver/Lu.h) carry the pivot tolerance
and factorization result. Compare the updated pivot magnitude with
`pivot_tolerance`; record `Success` on completion or `ZeroPivot` and its
zero-based `pivot_column` when a pivot is rejected. These records require no
permutation storage in this no-pivot algorithm.

## Toy example: factor and solve a 4-by-4 system

The input is a matrix A and right-hand side b. We want factors satisfying
$A=LU$ and a solution satisfying $Ax=b$:

```text
A = [4 2 2 2]     b = [22]     expected x = [1]
    [2 5 3 3]         [33]                  [2]
    [2 3 6 4]         [42]                  [3]
    [2 3 4 7]         [48]                  [4]
```

All indices are zero-based. First follow the row operations on paper; we will
connect them to the combined factor buffer after deriving L and U separately.

### 1. What are we trying to eliminate?

The first two equations of $Ax=b$ are

$$
\begin{aligned}
4x_0+2x_1+2x_2+2x_3&=22,\\
2x_0+5x_1+3x_2+3x_3&=33.
\end{aligned}
$$

We want to remove $x_0$ from the second equation. Subtracting one half of the
first equation does that because $2-(1/2)\cdot4=0$. The new second equation is

$$
0x_0+4x_1+2x_2+2x_3=22.
$$

This row operation preserves the solution: given the first equation, adding its
half back recovers the original second equation. To preserve the equation we
subtract the same multiple from every coefficient and from its right-hand side.
Factorization records the coefficient operations; the triangular solves later
apply their effect to b.

### 2. What is a multiplier, and why divide by the pivot?

At step k, the **pivot** is the current diagonal coefficient in row k. We use
that row to remove the coefficients below it in column k. A **multiplier**
$\ell_{ik}$ is how much of pivot row k to subtract from row i.

Let $a^{\mathrm{current}}_{ik}$ denote the coefficient after all earlier
elimination steps. To make this coefficient zero, choose the multiplier so that

$$
a^{\mathrm{current}}_{ik}-\ell_{ik}a^{\mathrm{current}}_{kk}=0
\quad\Longrightarrow\quad
\ell_{ik}=\frac{a^{\mathrm{current}}_{ik}}{a^{\mathrm{current}}_{kk}}.
$$

The ratio is chosen to cancel a specific coefficient. Check the pivot before
dividing. Here all pivots will be 4, so they pass the default zero tolerance.
Writing $R_i$ for the entire current coefficient row, the operation is

$$
R_i\leftarrow R_i-\ell_{ik}R_k.
$$

The target coefficient becomes zero. Every coefficient to its right changes
too, and those updated values determine the next elimination step.

### Step k matrix view

Let $A^{(k)}$ be the current coefficient matrix after columns before k have been
eliminated. The picture partitions it around the current pivot. The highlighted
column and row supply the two values needed to update each entry in the trailing
matrix. For now this is the mathematical matrix, with zeros below earlier
pivots; the multipliers are recorded separately in L.

```{=latex}
\begin{center}
\input{../solver/figures/dense-lu-step-k.tex}
\end{center}
```

For a particular row i and column j, divide the current column entry $c_i$ by
the pivot p to get $\ell_{ik}$. Then subtract $\ell_{ik}$ times the pivot-row
entry $r_j$ from $S_{ij}$. Repeating this for all $i,j>k$ updates the entire
trailing matrix. The expression $(\mathbf c/p)\mathbf r$ is an **outer product**:
a column times a row, producing a matrix with one update for each pair $(i,j)$.

In our 4-by-4 example, step $k=1$ has $p=4$, $\mathbf c=[2,2]^T$,
$\mathbf r=[2,2]$, and $S=\left[\begin{smallmatrix}5&3\\3&6\end{smallmatrix}\right]$.
The multipliers are $[1/2,1/2]^T$, so the outer product subtracts one from every
entry of S. This gives $\left[\begin{smallmatrix}4&2\\2&5\end{smallmatrix}\right]$,
exactly the remaining block after eliminating column 1 below.

### 3. Eliminate column 0

The pivot is 4. The entries below it are all 2, so
$\ell_{10}=\ell_{20}=\ell_{30}=2/4=1/2$. Subtract half of row 0 from each lower
row:

```text
R1: [2 5 3 3] - (1/2)[4 2 2 2] = [0 4 2 2]
R2: [2 3 6 4] - (1/2)[4 2 2 2] = [0 2 5 3]
R3: [2 3 4 7] - (1/2)[4 2 2 2] = [0 2 3 6]

current matrix = [4 2 2 2]
                 [0 4 2 2]
                 [0 2 5 3]
                 [0 2 3 6]
```

Column 0 is now finished. Keep the three multipliers separately for the moment.
The zeros in this displayed matrix are the actual eliminated coefficients.

### 4. Eliminate column 1 using the updated matrix

The new pivot is 4, and the two entries below it are 2. Thus
$\ell_{21}=\ell_{31}=2/4=1/2$. Use the current row 1:

```text
R2: [0 2 5 3] - (1/2)[0 4 2 2] = [0 0 4 2]
R3: [0 2 3 6] - (1/2)[0 4 2 2] = [0 0 2 5]

current matrix = [4 2 2 2]
                 [0 4 2 2]
                 [0 0 4 2]
                 [0 0 2 5]
```

Using the original A would give the incorrect ratio $3/5$ here. Both numerator
and denominator must include the changes from eliminating column 0. Row 1 has
a zero in column 0, so subtracting a multiple of it leaves that previously
eliminated column unchanged.

### 5. Eliminate column 2 to obtain U

The current pivot is 4. The only entry below it is 2, giving
$\ell_{32}=2/4=1/2$:

```text
R3: [0 0 2 5] - (1/2)[0 0 4 2] = [0 0 0 4]

U = [4 2 2 2]
    [0 4 2 2]
    [0 0 4 2]
    [0 0 0 4]
```

Check the last diagonal, 4. There are no rows below it to update. All entries
below the diagonal are zero, so this is an upper triangular matrix.

### 6. Why keep the multipliers in L?

U describes the equations after elimination. The multipliers tell us how the
original rows were built from these final rows. Reverse the subtractions by
adding back the recorded multiples:

$$
\begin{aligned}
A_{0,:}&=U_{0,:},\\
A_{1,:}&=\tfrac12 U_{0,:}+U_{1,:},\\
A_{2,:}&=\tfrac12 U_{0,:}+\tfrac12 U_{1,:}+U_{2,:},\\
A_{3,:}&=\tfrac12 U_{0,:}+\tfrac12 U_{1,:}+\tfrac12 U_{2,:}+U_{3,:}.
\end{aligned}
$$

For example, the original row 2 is recovered as

```text
(1/2)[4 2 2 2] + (1/2)[0 4 2 2] + [0 0 4 2] = [2 3 6 4].
```

Matrix multiplication expresses these row combinations as $A=LU$, with

```text
L = [ 1   0   0  0]
    [1/2  1   0  0]
    [1/2 1/2  1  0]
    [1/2 1/2 1/2 1]
```

The diagonal ones mean each original row contains one copy of its corresponding
U row. The entries below the diagonal are precisely the recorded multipliers.
Keeping them lets us apply the elimination to new right-hand sides without
repeating the factorization.

### 7. Forward substitution: solve Ly=b

Since $Ax=L(Ux)=b$, first find a vector y satisfying $Ly=b$; then solve $Ux=y$.
Each row of L gives one equation. Visit rows from first to last so that earlier
values of y are known:

$$
\begin{aligned}
y_0&=22,\\
y_1&=33-\tfrac12 y_0=33-11=22,\\
y_2&=42-\tfrac12 y_0-\tfrac12 y_1=42-11-11=20,\\
y_3&=48-\tfrac12 y_0-\tfrac12 y_1-\tfrac12 y_2=48-11-11-10=16.
\end{aligned}
$$

Thus y is `{22,22,20,16}`. These are the right-hand sides obtained by applying
the same elimination operations to b. L has diagonal ones, so no division is
needed in this substitution.

### 8. Backward substitution: solve Ux=y

The last equation contains only $x_3$. Solve it first, then move upward,
subtracting the contributions of unknowns already found:

$$
\begin{aligned}
x_3&=16/4=4,\\
x_2&=(20-2x_3)/4=(20-8)/4=3,\\
x_1&=(22-2x_2-2x_3)/4=(22-6-8)/4=2,\\
x_0&=(22-2x_1-2x_2-2x_3)/4=(22-4-6-8)/4=1.
\end{aligned}
$$

The result is the expected vector `{1,2,3,4}`. Checking the original equations
gives $Ax=\{22,33,42,48\}=b$, and multiplying the displayed L and U recovers A.

### 9. Map the mathematics to the factor buffer

Now that L and U are known, we can store them in one `DenseMatrix`. L's diagonal
is always one, and its entries above the diagonal are zero. U's entries below
the diagonal are zero. Store only their variable entries:

```text
combined factors.lu = [ 4   2   2  2]   diagonal and above: U
                      [1/2  4   2  2]   strictly below:    L
                      [1/2 1/2  4  2]
                      [1/2 1/2 1/2 4]

column-major buffer = {4,1/2,1/2,1/2, 2,4,1/2,1/2, 2,2,4,1/2, 2,2,2,4}
```

This combined matrix is a storage format for two factors. Its lower entries
hold the elimination history; the mathematical U still has zeros there.
Forward substitution reads those lower entries and supplies L's diagonal ones.
Backward substitution reads the diagonal and upper entries as U.

In `RightLookingFactorize`, start with `A.Clone()` so the original matrix remains available
for residual checks. At each pivot, store the multiplier in the slot of the
coefficient it eliminates, then update the entries to its right. The old
coefficient is no longer needed once its multiplier has been computed. After
every pivot passes its check, set `status.code=Success` and
`status.pivot_column=-1`. `Solve` uses a private `DenseVector` for y and leaves
both factors unchanged for another right-hand side.

## Implement in this order

Implement the routines in [`ScalarLU.cpp`](../../src/solver/dense/ScalarLU.cpp),
following the declarations in [`Lu.h`](../../include/solver/dense/Lu.h).
The containers are implemented already. Comment out a routine's placeholder
exception once its implementation is complete.

| Order | Function | Equation | Main idea |
|---|---|---|---|
| 1 | [`ForwardSolve`](../../src/solver/dense/ScalarLU.cpp) | L*y=rhs | Increasing rows; previous unknowns are ready; unit diagonal needs no division |
| 2 | [`BackwardSolve`](../../src/solver/dense/ScalarLU.cpp) | U*x=rhs | Decreasing rows; later unknowns are ready; check each U diagonal |
| 3 | [`RightLookingFactorize`](../../src/solver/dense/ScalarLU.cpp) | A=L*U | Copy input, form multipliers, update the trailing submatrix |
| 4 | [`Solve`](../../src/solver/dense/ScalarLU.cpp) | A*x=b | Reuse the factors for two substitutions and multiple RHSs |

For `ForwardSolve` and `BackwardSolve`, the tests supply known packed factors.
Do not call factorization inside either triangular solve. First reproduce y=`{22,22,20,16}` and
x=`{1,2,3,4}` from the toy example. Both kernels receive the `lu` matrix and
RHS/output vectors. The acceptance tests independently supply a 3-by-3 case
with different coefficients.

For `RightLookingFactorize`, use the scalar outer-product form of elimination. Before pivot k,
the remaining trailing submatrix includes every earlier pivot update. Check its
current diagonal before division, retain the new multipliers below it, and
update only the rows and columns after k. Print the trailing block in your own
debugging session and compare it with the hand calculation. No symbolic plan
or external numerical library is required.

For `Solve`, allocate a private intermediate y, use the completed triangular
solves, and preserve the factor buffers for another RHS. Known factors test
`Solve` independently. The integration test
`DenseLu.RightLookingFactorizeThenSolve` checks the complete pipeline after
the isolated checkpoints work.

### Compare scalar update orders

These routines update individual coefficients. They use the same packed L/U
format and pivot policy; only the timing and direction of updates differ.

| Function | Unit completed at each step | Where earlier work is applied |
|---|---|---|
| [`RightLookingFactorize`](../../src/solver/dense/ScalarLU.cpp) | Pivot column and trailing update | Immediately to all entries below and right of the pivot |
| [`LeftLookingFactorize`](../../src/solver/dense/ScalarLU.cpp) | One column of L and U | Gather earlier L-column contributions when the current column is reached |
| [`UpperLookingFactorize`](../../src/solver/dense/ScalarLU.cpp) | One row of L and U | Gather completed U-row contributions when the current row is reached |

`UpperLookingFactorize` names the row-oriented **up-looking** variant in this
library; see the [row- and column-oriented LU notes](https://www.cs.ubc.ca/~rbridson/courses/542g-fall-2008/notes-sept24.pdf).
The left-looking and up-looking bodies are exercises with implementation briefs.
After completing them, compare their reconstructed factors and solutions against
`RightLookingFactorize` on the same inputs. `Solve` works with all three results.

The [interactive HTML walkthrough](visuals/right-looking-lu.html) animates every
scalar update in the 4-by-4 example using the same packed `lu` buffer as the code.
To export the walkthrough as a captioned MP4
using Chrome and FFmpeg, run `make -C docs/solver video`. Temporary render files
stay in `docs/obj`; the video is written to `docs/video/right-looking-lu.mp4`.
Every video frame is retained as a 1920-by-1080 PNG screenshot in
`docs/obj/lu-video/frames/`; `capture-manifest.json` records the frame rate and count.
The renderer checks each packed-buffer operation and verifies `L*U=A` before export.

The matching [left-looking walkthrough](visuals/left-looking-lu.html) follows the
same matrix one column at a time, showing every gathered update, pivot check,
and multiplier. Run `make -C docs/solver video VIDEO_VARIANT=left-looking` to
create `docs/video/left-looking-lu.mp4`. Both HTML files have playback and timeline
controls; their videos include the explanations on screen, without audio.

The [8-by-8 symbolic walkthrough](visuals/right-looking-lu-8x8.html) keeps three
matrices on screen: teal L to the left of A, and blue U above A. KaTeX renders
every formula and matrix entry. Lines partition A before and after the current
pivot row and column. Yellow marks known factors, crossed terms mark removed
contributions, and highlights follow the current entry. Each scalar write adds
a persistent tilde; the active entry's superscript counts its updates. Run
`make -C docs/solver video VIDEO_VARIANT=right-looking-8x8` to create
`docs/video/right-looking-lu-8x8.mp4`; screenshots of every frame are retained in
`docs/obj/right-looking-lu-8x8-video/frames/`.

The matching [8-by-8 left-looking walkthrough](visuals/left-looking-lu-8x8.html)
uses the same colors, layout, and tilde transitions. It gathers completed source
columns into the current column, checks its pivot, and forms its multipliers;
future columns retain their original entries until visited. Run
`make -C docs/solver video VIDEO_VARIANT=left-looking-8x8` to create
`docs/video/left-looking-lu-8x8.mp4`. The renderer checks source coefficients,
column-local writes, untouched future columns, and agreement with the factors
from right-looking LU. PNG frames are saved in
`docs/obj/left-looking-lu-8x8-video/frames/`.

The [8-by-8 up-looking walkthrough](visuals/up-looking-lu-8x8.html) gathers
completed U rows into the current row. For each source row, it first divides
the gathered coefficient by that row's checked pivot to obtain an L multiplier,
then updates entries to its right. Future rows remain untouched. Run
`make -C docs/solver video VIDEO_VARIANT=up-looking-8x8` to create
`docs/video/up-looking-lu-8x8.mp4`; frame screenshots are saved in
`docs/obj/up-looking-lu-8x8-video/frames/`. The renderer verifies row-local
writes, source pivots, multiplier order, persistent tildes, and the final factors.

## Caller assumptions and numerical failures

These internal library routines assume that callers supply a valid square
matrix, host-accessible storage with a valid leading dimension, matching vector
sizes, and output buffers that do not overlap inputs. `RightLookingFactorize` assumes finite
input coefficients and a finite nonnegative `pivot_tolerance`; `Solve` assumes
successful factors. These are caller preconditions: do not write repeated shape,
pointer, overlap, overflow, or full-buffer finiteness checks in the LU exercises.
The existing containers handle their own allocation and view-construction checks.

- Copy A to independent owning factors and preserve the original input, including
  when A wraps a padded view. Padding is outside the mathematical matrix.
- Use no row swaps or pivot replacement. Reject a pivot if it is nonfinite or
  its magnitude is at most `pivot_tolerance`. Return `ZeroPivot` with its
  zero-based column before division. Partial factors are unusable by `Solve`.
- The forward kernel reads only strict-lower coefficients and assumes L's unit
  diagonal. The backward kernel reads only upper coefficients and rejects zero
  or nonfinite diagonals with `std::runtime_error`. These diagonal checks detect
  numerical failure before division; there is no separate coefficient scan.
- A successful empty factorization has a 0-by-0 matrix and permits empty
  substitutions/solves. The loops simply perform no work.
- Set `Success` and `pivot_column=-1` only after completing the factorization.
  Preserve status and numerical factors during solves.

Use SPD or strictly diagonally dominant examples for ordinary no-pivot success
checks. Failure tests deliberately include matrices outside these classes:
`[[0,1],[1,0]]` rejects column 0 despite being nonsingular, while
`[[2,4],[1,2]]` reaches a zero pivot at column 1. A tolerance equal to a pivot's
magnitude also rejects that pivot.

## Check and explain

Follow [the dense checkpoint commands](checkpoints.md#dense-checkpoints). The
acceptance executable is `test_solver_dense`, with CTest label `solver-dense`.
It checks the packed coefficients, L*U reconstruction, original-system residual,
input preservation, independent factor ownership, factor reuse, empty inputs,
and first/later/tolerance-boundary pivots. Malformed inputs and overlapping
outputs violate the caller preconditions and are outside these kernel tests.

Before moving on, explain these questions without reading the code:

1. Why do the two triangular solves traverse unknowns in opposite directions?
2. Why do diagonal entries in the packed buffer belong to U?
3. Why must the pivot test use the updated diagonal rather than A's original one?
4. Why do the multipliers for column 1 use 2/4 rather than the original 3/5?
5. Why is a new RHS cheaper than a new factorization?
6. How do L*U reconstruction and the original-system residual check different outputs?

Then continue to [scalar sparse LU](scalar-lu.md).
