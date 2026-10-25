# Serial supernodal LU

Goal: perform elimination using dense blocks inside a sparse structure. Reuse
the scalar stage's mathematical conventions, but implement a real block factorization.

Implement W2.1–W2.5 in `src/solver/supernodal/Lu.cpp`. The supplied plan partitions
ordered columns into supernodes and lists every source panel read by each node.
See `include/solver/supernodal/Lu.h` for exact buffer layouts.

Each panel block is a `DenseMatrix`: diagonal w-by-w, lower r-by-w, and upper
w-by-c. Use its two-dimensional access and leading dimension; the row/column
maps still translate between local entries and global ordered coordinates.

Before coding, derive the block equations in [foundations](foundations.md) and
trace [the dense cross](worked-example.md#2-one-dense-cross-diagonal-lu-and-two-triangular-solves).
Use the [isolated block tests](checkpoints.md#supernodal-checkpoints) in task order.

## Tasks

1. **W2.1 `AssemblePanel`**: build a panel's dense cross from the original matrix
   using the symmetric permutation. Zero all blocks before scattering. The cross
   contains A11, A21 below it, and A12 to its right; do not materialize a dense n*n
   trailing matrix. Check block-size arithmetic in OT before allocation.
2. **W2.2 `UpdatePanel`**: for each completed contributor, intersect its row and
   column maps with this cross, then apply `A(R,C) -= L(R,K)*U(K,C)`. Update all
   three cross blocks where relevant. Missing structural couplings mean zeros;
   indices are ordered matrix coordinates, not positions in another panel's
   buffers. Keep all temporary products private to the workspace.
3. **W2.3 `FactorPanel`**: compute no-pivot LU of A11. Store strict-lower L11 and
   upper U11 in the same diagonal buffer, with L11's unit diagonal implicit.
   Compute L21 from `L21*U11=A21` and U12 from `L11*U12=A12`. Move completed
   storage into the output panel. Report a rejected pivot using its global
   ordered column, not its offset inside A11.
4. **W2.4 `Factorize`**: validate the scalar and block plans and traverse panels
   serially. Panels must partition [0,n); row/column lists are sorted, unique,
   in range, and beyond the panel; contributor IDs are unique and earlier.
   The instructor guarantees structural closure and dependency completeness.
   Call assembly, updates, and panel factorization as distinct stages.
5. **W2.5 `Solve`**: implement serial block forward/backward substitution using
   L11/L21 and U11/U12, with the same permutation convention as the scalar stage. Remember
   that the packed diagonal contains U's diagonal, not L's implicit ones.

Begin with readable dense loops. Then use single-threaded BLAS for suitable
TRSM/GEMM operations if available. Dense BLAS is allowed inside panels; a dense
whole-matrix factorization and scalar-factor-then-pack implementation are not.

## Checks and submission

Run `test_solver_supernodal` / CTest label `solver-supernodal`. Supplied tests include
a width-two panel, separate assembly/update/factor checks, a later failed pivot,
and a nonsymmetric case with different lower/upper maps and a permuted solve.
Add a width-one partition comparison, a diagonal block of width greater than
two, malformed panel bounds, and tolerance-boundary failures. Compare
against the scalar stage or an independent reference with tolerances, not bitwise equality.

Submit code and a short report with:

- A diagram of the panel cross, its dimensions, and its column-major buffers.
- The three update shapes and the two block triangular-solve equations.
- A scalar/supernodal correctness comparison using the same ordered matrix.
- Timing and memory observations for instructor-supplied larger plans, including
  the effect of panel width. Identify time spent gathering/scattering separately
  from useful dense arithmetic.

Follow [source-guide lessons C and D](source-reading.md#lesson-c-supernodes-panels-and-dense-arithmetic).
Explain how a structural supernode differs from a numerical panel, how PaStiX
describes blocks, and why dense work can help despite storing some zeros. Draw
the actual leading dimensions before comparing BLAS calls or transpose flags.
