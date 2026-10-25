# Scalar sparse LU and triangular solves

Goal: explain and implement sparse left-looking elimination after deriving basic
LU in [Stage 0](README.md#stage-0-understand-lu-before-implementing-sparse-lu).
Complete `test_solver_dense` first. The dense factors provide an independent
small-case comparison while you learn CSC traversal and sparse accumulation.
Symbolic analysis and ordering are supplied so the first sparse implementation
stays numerical.

Implement TODOs W1.1–W1.5 in `src/solver/scalar/Lu.cpp`, following
`include/solver/scalar/Lu.h` and the [shared assignment contract](README.md).
Keep the public signatures and existing matrix ownership rules.

## Tasks

Follow this implementation order; W numbers identify exercises, not prerequisites.
Use the [direct checkpoint tests](checkpoints.md#scalar-checkpoints) before
integrating the driver.

1. **W1.3 `ForwardSolve`**: use the unit-lower CSC factors to solve in increasing
   column order. Account for the explicitly stored diagonal.
2. **W1.4 `BackwardSolve`**: solve with U in decreasing column order and check its
   diagonal before division.
3. **W1.1 `FactorColumn`**: scatter one ordered column into the sparse
   accumulator. Use the known U pattern to visit earlier pivot columns in
   increasing order, save each U coefficient, and apply the corresponding
   completed L column to remaining rows. Inspect the resulting pivot, then store
   the unit L diagonal and normalized entries below it. Preserve planned entries
   even when their numerical value cancels to zero. Reset touched scratch entries.
4. **W1.2 `Factorize`**: validate the CSC input and supplied plan, allocate L/U
   with those exact patterns, construct inverse-permutation scratch once, and
   invoke W1.1 in increasing column order. Return `Success` only after every
   column succeeds; return `ZeroPivot` before an invalid division.
5. **W1.5 `Solve`**: gather b into ordered coordinates, call the two triangular
   solves, and scatter x into original coordinates. Do not overwrite b. Reuse the
   same factors for a second RHS.

CSC inputs must be square with finite values, valid monotone offsets, in-range
sorted unique row IDs, and storage for each declared entry. Symbolic L/U patterns
must have the right dimensions, triangular indices, and a diagonal in each
column. Validate a nonempty permutation as a bijection. The instructor guarantees
that the supplied pattern contains all fill; do not perform symbolic elimination
just to re-prove that guarantee. Overflow-check sizes and narrowing conversions.

For each function, first reason through the small valid-input arithmetic, then
complete its required validation and failure paths before moving on. Keep full
reset in the driver distinct from per-column cleanup: `Resize` resets the
inverse permutation to identity, so the partial call in `FactorColumn` needs
attention when you implement it. The permuted-column checkpoint preserves a
nonidentity inverse map across all columns.

## Checks and submission

Run `test_solver_scalar` / CTest label `solver-scalar`. Direct tests of both
triangular solves and column factorization are supplied. Add a dimension mismatch,
a missing diagonal in the plan, and failure-path scratch cleanup. Retain the supplied
permutation, fill, empty, and pivot-failure cases.

Submit your code and a short report containing:

- A hand-worked four-cycle example identifying the new fill entry.
- Your accumulator invariant before and after applying a previous L column.
- The meaning of `order` and the two vector permutations.
- Factorization and solution errors, plus sparse storage size.
- Why `[[0,1],[1,0]]` is nonsingular yet fails this no-pivot algorithm.

Follow [source-guide lessons A and B](source-reading.md#lesson-a-drivers-and-triangular-solves)
after the matching kernels work. Explain which structural work the plan replaces.
Keep the detailed `pdgstrf_pivotL.c` study for Stage 5, when you can contrast it
with a working no-pivot factorization. DFS, pruning, allocator, and pivot
machinery are reading topics outside the W1 implementation.
