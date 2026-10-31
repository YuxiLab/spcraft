# Predict, implement, test, explain

These checkpoints follow [the implementation sequence](README.md).
The tests are targets for student code: the unfinished numerical functions still
throw placeholder exceptions. Expected failures are not skipped or converted
to passes. Test discovery or successful compilation does not establish correctness.

## Prepare the local build

Use the dependency setup described in the [assignment README](README.md).
For a CPU-only build, a typical local configuration is:

```bash
cmake -S . -B build/solver-learning \
  -DSPCRAFT_BUILD_DOWNSTREAM=ON -DBUILD_TESTING=ON \
  -DSPCRAFT_BUILD_SOLVER_ASSIGNMENTS=ON \
  -DSPCRAFT_USE_MPI=OFF -DSPCRAFT_USE_CUDA=OFF -DSPCRAFT_USE_MKL=OFF \
  -DSPCRAFT_BUILD_EXAMPLES=OFF -DSPCRAFT_BUILD_BENCHMARKS=OFF
cmake --build build/solver-learning \
  --target test_dense_matrix test_solver_dense test_solver_scalar test_solver_supernodal test_solver_mt --parallel 4
```

For the parallel stage, ensure CMake found OpenMP; on macOS the repository's libomp setup
may require `OpenMP_ROOT`. A serial build is useful to check fallback behavior,
but does not verify multithreaded execution. On remote servers, submit builds,
tests, and benchmarks through Slurm compute nodes.

Check the teaching arithmetic independently of C++ student implementations:

```bash
python3 tests/solver/verify_learning_examples.py
ctest --test-dir build/solver-learning -R '^dense_matrix\.' --output-on-failure
build/solver-learning/bin/test_solver_scalar --gtest_filter='LearningFixtures.*:ScalarWorkspace.*'
```

The Python checker uses exact fractions and supplied answers; it checks products,
traces, and coordinate mappings, not an implementation of LU. `LearningFixtures`
checks the nonsymmetric fixture and its known factors. `ScalarWorkspace` checks
the existing scratch initialization. These can pass before any W exercise works.

## Extension checkpoints after the core

Finish the current core targets first. The [extension guide](extensions.md#checkpoints)
lists commands for the separate `test_solver_extensions` target. Its independent
fixture test passes immediately; analysis, pivoting, solve, and refinement checks
remain acceptance exercises. Each new function has an active placeholder.

```bash
cmake --build build/solver-learning --target test_solver_extensions
build/solver-learning/bin/test_solver_extensions --gtest_filter='ExtensionFixtures.*'
```

## Dense checkpoints

Begin here after the hand calculation. The [dense chapter](dense-lu.md) defines
the packed layout, caller assumptions, and numerical failures; its declarations
are in `solver/dense/Lu.h`.
Matrix dimensions and stride belong to `DenseMatrix`; only RHS/solution storage
uses `DenseVector`. The container tests pass before the LU exercises are implemented.

```bash
build/solver-learning/bin/test_solver_dense --gtest_filter='W01DenseForwardSolve.*'
build/solver-learning/bin/test_solver_dense --gtest_filter='W02DenseBackwardSolve.*'
build/solver-learning/bin/test_solver_dense --gtest_filter='DenseRightLookingFactorize.*'
build/solver-learning/bin/test_solver_dense --gtest_filter='W04DenseSolve.*'
ctest --test-dir build/solver-learning -L solver-dense --output-on-failure
```

| Order | Predict before running | Explain before moving on |
|---|---|---|
| W0.1 | y=`{9,15/2,55/3}` from the supplied packed factors and b | Why is the stored diagonal ignored? |
| W0.2 | x=`{1,2,3}` from supplied y | Why do later unknowns have to be ready first? |
| W0.3 | Packed LU=`{4,1/2,0,1,9/2,2/9,1,-1/2,55/9}` | Which entries are multipliers, and which trailing entries are updated? |
| W0.4 | Original-system residual is small for two RHSs | Why is only substitution repeated, rather than factorization? |

The two triangular suites use hand-supplied packed factors. `W04DenseSolve`
also uses supplied factors, so it can be tested before integrating W0.3.
`DenseLu.RightLookingFactorizeThenSolve` is the final integration check. The empty-input,
padded-storage, pivot, and tolerance cases check the remaining dense behavior.
The caller supplies valid shapes and disjoint buffers; the kernels need no
repeated input validation.
Before proceeding, explain how `i+j*n` describes column-major storage and why
the lower multipliers are retained instead of replaced by eliminated zeros.
Padded views use `i+j*ld`; `A[i,j]` handles this distinction automatically.

Additional useful checks include a nonfinite pivot produced by arithmetic
overflow from finite inputs and the exported wider index/offset types.
After this stage passes, keep its tests while adding sparse LU.

## Scalar checkpoints

First check the supplied sparse-column infrastructure and example factors:

```bash
build/solver-learning/bin/test_solver_scalar --gtest_filter='SparseColumnWorkspace.*:SparseColumnFixtures.*'
```

Then implement and check each phase independently. These tests supply their own
inputs and factor storage, so no other unfinished numerical phase is needed:

```bash
build/solver-learning/bin/test_solver_scalar --gtest_filter='SparseScatterColumn.*'
build/solver-learning/bin/test_solver_scalar --gtest_filter='SparseUpdateColumn.*'
build/solver-learning/bin/test_solver_scalar --gtest_filter='SparseStoreColumn.*'
```

The [4-by-4 sparse example](scalar-lu.md#toy-example-fill-in-a-4-by-4-matrix)
shows the expected accumulator `{1,15/4,-1/4,-1/4}` after updating column 1.
The update test poisons unfinished factor values with NaNs: only completed L
columns and the current U column's planned row indices may be read.
`StoreColumn` returns a column status without changing the driver-owned status
or clearing scratch. `FactorColumn` owns cleanup on every exit.

Build `test_solver_scalar` after each change. Run just the selected exercise:

```bash
build/solver-learning/bin/test_solver_scalar --gtest_filter='W13ForwardSolve.*'
build/solver-learning/bin/test_solver_scalar --gtest_filter='W14BackwardSolve.*'
build/solver-learning/bin/test_solver_scalar --gtest_filter='W11FactorColumn.*'
```

| Order | Predict before running | Explain before moving on |
|---|---|---|
| W1.3 | Worked-example b produces y=`{9,15/2,55/3}` | Why is y[j] ready, and why is no division needed? |
| W1.4 | The supplied U and y produce x=`{1,2,3}` | Why must larger columns finish first? |
| W1.1 | Nonsymmetric factors reconstruct B in both orders; U(1,2)=0 remains stored | Which rows are touched through fill, and why must inverse_order survive reuse? |
| W1.2 | Owning factors contain the exact supplied patterns | Which checks precede access/allocation, and which correctness properties are supplied by the plan? |
| W1.5 | Original x is recovered after gathering and scattering; a second RHS reuses factors | Which vector is ordered, and which vector is in original coordinates? |

`W11FactorColumn` allocates buffers and initializes the inverse map in the test;
it does not need `Factorize`. The test also checks clean scratch and that the
column kernel leaves the overall `factors.status` to its driver.

When W1.2 is ready but W1.5 is unfinished, use the pivot/empty validation tests
as appropriate or write a factor-only experiment; the integrated success tests
also call `Solve`. Once the whole scalar stage works, run:

```bash
build/solver-learning/bin/test_solver_scalar
ctest --test-dir build/solver-learning -L solver-scalar --output-on-failure
```

Additional student checks: missing diagonals, overlapping vectors, invalid CSC
pointers/rows, nonfinite coefficients/tolerance, empty input, nonfinite pivots,
tolerance boundaries, cancellation, and scratch cleanup on pivot failure. The
supplied direct tests are deliberately focused; they are not a complete grader.

## Supernodal checkpoints

Each direct test constructs its inputs without calling another unfinished kernel:

```bash
build/solver-learning/bin/test_solver_supernodal --gtest_filter='W21AssemblePanel.*'
build/solver-learning/bin/test_solver_supernodal --gtest_filter='W22UpdatePanel.*'
build/solver-learning/bin/test_solver_supernodal --gtest_filter='W23FactorPanel.*'
```

| Order | Predict before running | Explain before moving on |
|---|---|---|
| W2.1 | First nonsymmetric cross is `{8}`, `{2,4}`, `{2}` in both coordinate orders | Why does the second lower entry represent global row 3? |
| W2.2 | The two sources change the last diagonal cross to column-major `{12,3,1,65/4}` | How do all three destination products use source maps and absent couplings? |
| W2.3 | The 2-by-2 worked cross yields packed `{4,1/2,2,3}`, lower `{1,2}`, upper `{2,1}` | Why do the two triangular solves use different sides and diagonal policies? |
| W2.4 | Block reconstruction matches the same ordered B as scalar LU | Why is a scalar factorization followed by packing a different algorithm? |
| W2.5 | Original x and a doubled RHS solution match the scalar stage | Which diagonal values belong to U, and where is L's unit diagonal? |

The update tests cover all three products, nonconsecutive maps, a missing source
coupling, an inner dimension greater than one, and accumulation from two sources.
The pivot test fails after the first local column and expects global column 6.
The nonsymmetric integration test includes unequal R/C maps and stored cancellation.

When the stage is complete:

```bash
build/solver-learning/bin/test_solver_supernodal
ctest --test-dir build/solver-learning -L solver-supernodal --output-on-failure
```

Add malformed block bounds/lengths, width-one versus wider partitions, a panel
wider than two, empty contributors/off-diagonal blocks, tolerance boundaries,
and solve validation/failure cases. Record your choice of tolerances and error
measures; do not demand bitwise agreement between different arithmetic orders.

## Parallel checkpoints

```bash
build/solver-learning/bin/test_solver_mt --gtest_filter='MtLu.ContributorGraphAndInvalidEdges'
build/solver-learning/bin/test_solver_mt --gtest_filter='MtLu.IndependentBranchesJoinBeforeParent'
build/solver-learning/bin/test_solver_mt --gtest_filter='MtLu.RepeatedNonsymmetricFactorization'
ctest --test-dir build/solver-learning -L solver-mt --output-on-failure
```

For W3.1, predict `successor_ptr={0,2,3,3}`, `successor_id={1,2,2}`, and
`indegree={0,1,2}` for the original block fixture. The fork/join numerical test
is a different graph: `0->2<-1`. Explain why these examples have different
available concurrency.

For W3.2, first use one worker and prove normal completion. Then use several
workers, prove publication ordering, and add stop/drain behavior. The repeated
nonsymmetric test checks fresh factorization state across calls and thread counts;
the rejected-pivot test checks returning without deadlock. CTest supplies a
30-second per-test timeout; use it for failure-path runs so a deadlock is bounded.

Add empty graphs, duplicate/self/forward edges, zero/negative requested threads,
several independent branches, captured worker exceptions, and independent
concurrent numerical failures. Correct numerical results alone cannot prove
race freedom. Explain ownership and synchronization, and use a race detector
when supported by your compiler/OpenMP runtime.

## Completion evidence

Retain the previously passing stages. The added fixtures use `(IT,OT)=(32,32)`;
also exercise the exported `(32,64)` and `(64,64)` combinations before declaring
the full assignment complete. For every completed stage, record factor residual,
original-system backward error, input preservation, and factor reuse. Performance
experiments need larger valid plans and repeated timings; these tiny arithmetic
fixtures establish numerical behavior, not speedup.

After all four implementation stages, complete source-guide lessons F–H and the Stage 4–6
exit evidence. Being able to trace a routine, its arrays, and its invariants is
the goal; recognizing a routine's name is only the starting point.
