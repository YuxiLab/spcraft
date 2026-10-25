# Solver learning and implementation guide

Build one solver in four implementation stages: dense LU, scalar sparse
elimination, serial supernodal factorization, and shared-memory panel scheduling.
Start with the LU foundations, then use the completed kernels to study the structure,
pivoting, and runtime choices in SuperLU_MT and PaStiX. The C++ assignment directories
contain interfaces and TODO stubs, **not reference implementations**. Calling
an unfinished function throws `std::logic_error` with its assignment ID.

This README is the main guideline for the solver chapters. Follow the sequence
below; assignment IDs do not determine implementation order.
For a live task list with source navigation, use the included
[Neovim solver-progress plugin](../../nvim/solver-progress.nvim/README.md).

| Stage | Implementation | Instructions | Acceptance target |
|---|---|---|---|
| 0 | `include/solver/dense/`, `src/solver/dense/` | [Dense LU](dense-lu.md) | `test_solver_dense` |
| 1 | `include/solver/scalar/`, `src/solver/scalar/` | [Scalar LU](scalar-lu.md) | `test_solver_scalar` |
| 2 | `include/solver/supernodal/`, `src/solver/supernodal/` | [Supernodal LU](supernodal-lu.md) | `test_solver_supernodal` |
| 3 | `include/solver/mt/`, `src/solver/mt/` | [Parallel LU](parallel-lu.md) | `test_solver_mt` |

The stages are milestones, not deadlines. Learning C++, CSC storage, and
OpenMP at the same time requires additional preparation. Work at a pace that
lets you explain each checkpoint before continuing.

## How to use this course

1. Read [LU foundations](foundations.md), then work through the
   [numerical examples](worked-example.md) on paper.
2. Implement one function at a time in the sequence below.
   Predict its result before running the corresponding [checkpoint test](checkpoints.md).
3. Explain the invariant and draw the storage that function reads and writes.
   Complete the numerical behavior and required failure handling. The internal
   dense kernels assume valid caller-supplied shapes and buffers; their exercises
   need pivot checks, without repeated input validation.
4. At each stage, follow the matching [source-reading lesson](source-reading.md).
   Identify both the shared idea and the additional problem the library solves.
5. After the parallel solver works, complete the symbolic-analysis, pivoting,
   and refinement studies at the end of the sequence. These explain important
   concepts deliberately removed from the first implementation.

Follow this checklist from top to bottom. Assignment IDs identify the original
tasks; they do not have to match implementation order. First derive a small LU
factorization, then implement the dense triangular solves and dense LU. Translate
those ideas to sparse storage only after the dense stage works.
Each function body contains numbered comments describing its stages.
Replace its final TODO exception with your implementation as you complete it.

Read the [shared contract](README.md#mathematical-scope) and
[`include/solver/Lu.h`](../../include/solver/Lu.h) first. The instructor supplies
ordering and symbolic plans; you implement numerical kernels and scheduling.
Keep the public signatures, owning matrix/vector types, and Boost.Preprocessor
instantiations. The compiled combinations use `double` with `(IT,OT)` widths
`(32,32)`, `(32,64)`, and `(64,64)`.

## Stage 0: understand LU before implementing sparse LU

Read [foundations](foundations.md) and [the worked examples](worked-example.md).
No production-solver source reading is needed yet.

- [ ] Derive the multipliers and Schur update in the 3-by-3 example. Multiply
  the supplied L and U to check them; explain the new fill entry.
- [ ] Solve for the known x using two triangular solves. Explain why L's diagonal
  is one and why U's diagonal must be checked before division.
- [ ] Explain why a zero pivot can occur in a nonsingular matrix and why a small
  residual alone does not establish a small forward error.

Then complete [the dense LU exercises](dense-lu.md) in
`src/solver/dense/ScalarLU.cpp`. Work with one column-major buffer, without CSC,
symbolic analysis, or permutations. The packed lower multipliers and upper
triangle use the same convention as the later diagonal panel buffer.

The implemented `DenseMatrix` class owns dimensions and column-major storage;
use C++23 indexing `A[row,col]` and `A.ld` to connect matrix notation to physical
layout. The existing `A(row,col)` syntax is also available.
Right-hand sides and solutions remain `DenseVector`s. Check the container with
`test_dense_matrix` before working on the numerical exercise bodies.

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 1 | `dense::ForwardSolve` / W0.1 | Unit-lower substitution and column-major indexing | Known packed L produces y=`{9,15/2,55/3}`; ignore U's stored diagonal. |
| 2 | `dense::BackwardSolve` / W0.2 | Upper substitution and reverse traversal | Known packed U produces x=`{1,2,3}`; reject zero/nonfinite diagonals. |
| 3 | `dense::RightLookingFactorize` / W0.3 | Multipliers and the trailing Schur update | Verify packed factors, L*U=A, input preservation, empty input, and rejected pivots. |
| 4 | `dense::Solve` / W0.4 | Both dense triangular solves | Reuse supplied factors for two RHSs and check b-A*x in original coordinates. |

Then compare the scalar update orders by completing `dense::LeftLookingFactorize`
and `dense::UpperLookingFactorize` in the same source file. Their briefs describe
column-oriented left-looking and row-oriented up-looking LU. They return the
same factor format for `dense::Solve`.

- [ ] Pass the four [dense checkpoints](checkpoints.md#dense-checkpoints), then
  `test_solver_dense` / `solver-dense`.
- [ ] Explain why a packed diagonal belongs to U and L's unit diagonal is implicit.
- [ ] Translate the worked example to CSC and locate column 1 using its offsets.
- [ ] Gather and scatter with `order={2,0,1}` without confusing the inverse map.

Dense LU is a required first implementation, not a backend for the sparse stages.
Keep W1/W2 numerical work in the prescribed sparse/block algorithms.

Exit evidence: passing dense checkpoints, the elimination trace, packed L/U,
and the CSC/permutation translation. Run the arithmetic checker in [checkpoints](checkpoints.md)
after calculating your own answers.

## Stage 1: scalar sparse LU

Work in [`src/solver/scalar/Lu.cpp`](../../src/solver/scalar/Lu.cpp), following
[`include/solver/scalar/Lu.h`](../../include/solver/scalar/Lu.h) and
[scalar LU instructions](scalar-lu.md).

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 5 | `scalar::ForwardSolve` / W1.3 | CSC column traversal and unit-lower structure | Hand-built L: verify `L*y=rhs`, explicit diagonal handling, unchanged RHS, and empty input. |
| 6 | `scalar::BackwardSolve` / W1.4 | CSC column traversal and upper structure | Hand-built U: verify `U*x=rhs`, reverse traversal, and rejection of zero/nonfinite diagonals. |
| 7 | `scalar::FactorColumn` / W1.1 | Ordered coordinates, supplied L/U patterns, accumulator | With manually allocated patterns and scratch, factor columns sequentially; verify a column with fill, stored cancellation zeros, and scratch cleanup on success/failure. |
| 8 | `scalar::Factorize` / W1.2 | `FactorColumn` | Validate/allocate once, run the column loop, and check `P^T*A*P=L*U`, malformed plans, empty input, and rejected pivots. |
| 9 | `scalar::Solve` / W1.5 | `ForwardSolve`, `BackwardSolve`, successful factors | Check a nonidentity permutation, original-coordinate residual, unchanged inputs, and reuse for a second RHS. |

For the invariant, expected values, and commands for each row, use
[scalar checkpoints](checkpoints.md#scalar-checkpoints). After W1.3/W1.4, read
lesson A in [the source guide](source-reading.md#lesson-a-drivers-and-triangular-solves).
After W1.1/W1.2/W1.5, read lesson B. Keep production pivoting for Stage 5.

- [ ] Implement and directly check `ForwardSolve`.
- [ ] Implement and directly check `BackwardSolve`.
- [ ] Implement `FactorColumn` using preallocated factors and reusable scratch.
- [ ] Implement `Factorize` and verify factor reconstruction.
- [ ] Implement `Solve` and pass `test_solver_scalar` / `solver-scalar`.

The driver allocates the exact symbolic patterns and constructs `inverse_order`
once. `FactorColumn` performs numerical work on one column; it does not discover
fill or allocate new factor entries. During elimination, the accumulator holds
the scattered ordered column minus contributions already applied. Record each
touched accumulator location once, including locations reached through fill,
and clear it before the workspace is reused.

`Workspace::Resize` initializes a new factorization and resets `inverse_order`
to identity. The current partial W1.1 body calls it on every column. When
implementing W1.1, reconcile that call with the once-built permutation: retain
the initialized inverse map across columns and clear only numerical scratch.
The direct permuted-column test detects an accidental identity reset.

Both factors are in ordered coordinates:

```text
order[new] = old
B(i,j) = A(order[i], order[j]) = (L*U)(i,j)
rhs[i] = b[order[i]]
L*y = rhs; U*z = y
x[order[i]] = z[i]
```

An empty `order` means identity. Use a no-pivot input class from the shared
contract and report rejected pivots before division; do not modify the pivot.

## Stage 2: serial supernodal LU

Work in [`src/solver/supernodal/Lu.cpp`](../../src/solver/supernodal/Lu.cpp), using
the layouts in [`include/solver/supernodal/Lu.h`](../../include/solver/supernodal/Lu.h)
and [supernodal LU instructions](supernodal-lu.md). Finish the scalar stage first so its factors and solves
can serve as an independent numerical comparison.

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 10 | `supernodal::AssemblePanel` / W2.1 | Permutation mapping and column-major blocks | Compare A11/A21/A12 with a small ordered matrix; verify zero initialization and reuse of workspace for different panel sizes. |
| 11 | `supernodal::UpdatePanel` / W2.2 | `AssemblePanel`; hand-built completed contributor factors | Verify all three update products against a dense calculation, including multiple contributors, missing couplings, and distinct row/column maps. |
| 12 | `supernodal::FactorPanel` / W2.3 | Updated cross and dense no-pivot elimination | Check `A11=L11*U11`, `A21=L21*U11`, and `A12=L11*U12`; reject a later local pivot using its global column. |
| 13 | `supernodal::Factorize` / W2.4 | All three panel kernels | Validate the partition, run the serial pipeline, and reconstruct the factors for width-one and wider panels. |
| 14 | `supernodal::Solve` / W2.5 | Successful block factors and triangular substitution | Match the scalar solution with a nonidentity permutation, implicit L diagonal, empty off-diagonal blocks, and repeated RHS solves. |

Use [block checkpoints](checkpoints.md#supernodal-checkpoints) to test assembly,
updates, and local factorization without a finished driver. Read source-guide
lesson C after the three kernels work, and lesson D after the block solve works.

- [ ] Implement `AssemblePanel` and check each gathered block.
- [ ] Implement `UpdatePanel` with manually constructed source panels.
- [ ] Implement `FactorPanel` and check the three factor equations.
- [ ] Implement `Factorize` and compare with scalar LU.
- [ ] Implement `Solve` and pass `test_solver_supernodal` / `solver-supernodal`.

For one destination panel, let `J=[first,last)`, `R=rows`, and `C=columns`.
The workspace cross contains `A11=B(J,J)`, `A21=B(R,J)`, and `A12=B(J,C)`.
For each completed contributor with columns `K`, subtract:

```text
A11 -= L(J,K) * U(K,J)
A21 -= L(R,K) * U(K,J)
A12 -= L(J,K) * U(K,C)
```

Then factor A11 and solve for L21/U12. Work inside these blocks; do not allocate
or factor a dense whole matrix, or factor scalar CSC and pack it afterwards.
For tests of `UpdatePanel`, construct completed source panels directly so you
can isolate mapping and update mistakes before integrating the full driver.

## Stage 3: multithreaded factorization

Work in [`src/solver/mt/Lu.cpp`](../../src/solver/mt/Lu.cpp), following
[`include/solver/mt/Lu.h`](../../include/solver/mt/Lu.h) and
[parallel LU instructions](parallel-lu.md). Finish and verify the serial block kernels first.

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 15 | `mt::BuildTaskGraph` / W3.1 | Complete contributor lists and CSR prefix sums | Hand-check successor ranges/indegrees for a fork/join graph; test empty input, duplicates, and invalid/self/forward edges. |
| 16 | `mt::Factorize` / W3.2 | `BuildTaskGraph` and the supernodal numerical kernels | Match serial factors/solutions, then test 1/2/4/8 threads, repeated runs, independent branches, and pivot/exception paths without deadlock. |

Use [parallel checkpoints](checkpoints.md#parallel-checkpoints), then read
source-guide lesson E. Draw the fork/join graph before writing worker code.

- [ ] Implement `BuildTaskGraph` and check its CSR arrays directly.
- [ ] Implement `Factorize` first with one worker using the ready queue.
- [ ] Enable multiple workers with private workspaces and synchronized publication.
- [ ] Add stop/drain handling for numerical failures and worker exceptions.
- [ ] Pass `test_solver_mt` / `solver-mt` and run the additional concurrency checks.

Preallocate all output slots before launching workers. A panel becomes ready
only after **every** listed contributor has successfully published its data.
One worker owns each active panel's writes; source factors are immutable after
publication. Protect the ready queue and dependency counters, and establish
visibility of panel writes before another worker reads them. Perform numerical
kernels outside the scheduler lock. An empty queue while work is active means
workers must wait; it does not mean factorization is complete.

On failure, stop releasing new tasks, drain active workers, and join before
returning unsuccessful factors or rethrowing a captured exception. Reuse
`supernodal::Solve`; a new parallel solve is outside the assignment.

## Stage 4: learn what the supplied symbolic plan hides

Prerequisite: the serial block solver works. Use a separate scratch program or
notebook for these experiments; the required solver API continues to consume an
immutable supplied plan. Follow source-guide lesson F.

1. Write a small separate symbolic-elimination program using sets of structural
   positions. On the 3-by-3 example, for each
   pivot, connect its lower rows to its upper columns. Record new positions
   even if a numerical calculation would cancel them to zero. Compare the
   resulting triangular patterns with the worked example.
   Expect five L slots and six U slots, including both stored diagonals. Counting
   the diagonal once gives eight factor positions versus seven input entries.
2. Repeat on a small grid in natural order and a hand-chosen separator order.
   Count factor slots and estimate scalar updates. Explain why reordering
   changes fill while representing the same original linear system.
3. Group compatible consecutive columns into fundamental supernodes. Describe
   how merging additional columns can store zeros yet expose larger dense
   kernels. Distinguish a structural supernode from a chosen panel width.
4. Draw an elimination tree for a small symmetric-pattern case and the complete
   contributor DAG used by your code. Explain how a tree can summarize ancestry
   while the numerical kernels still require more read dependencies.

Exit evidence: input/fill patterns, two ordering comparisons, a proposed block
partition, and a map from `LuSymbolic`/`SupernodalSymbolic` to the library analysis
data. A small set-based symbolic experiment is sufficient; production ordering
and symbolic algorithms are reading topics, not new requirements for W1–W3.

## Stage 5: learn why the libraries use different pivot policies

Prerequisite: you can explain every no-pivot rejection and keep original,
ordered, and local coordinates distinct. Follow source-guide lesson G.

1. Work by hand with `[[0,1],[1,0]]` and `[[epsilon,1],[1,1]]` for positive small
   epsilon. Track the multipliers with and without a row interchange. The
   second matrix lies outside the core assignment's safe input classes.
   Then write a separate dense 2-by-2 comparison experiment. With `epsilon=1e-8`,
   no row swap gives multiplier `1e8`; swapping gives multiplier `1e-8`.
   A relative threshold of 0.1 selects the larger pivot in the first column.
   Replacing the first diagonal by `1e-4` instead gives multiplier `1e4` and
   changes the factorized matrix. Check each policy against the original A.
2. Trace SuperLU_MT's threshold comparison and its updates to the row mapping.
   Write the separate row and column permutation equation `Pr*A*Pc=L*U` using
   that library's convention; derive the RHS and solution mappings from it.
3. Trace PaStiX's diagonal replacement and pivot count. Explain which matrix is
   effectively factorized when entries are changed, and why checking the
   residual against the original A remains necessary.
4. Compare rejection, row pivoting, and static replacement in a short table.
   List which fixed storage and ownership assumptions would need reconsidering
   before adding dynamic row pivoting to this solver.

Exit evidence: two numerical traces and a design note. Dynamic pivoting is an
architectural extension, not a change confined to `pivot_tolerance`. Do not
silently insert replacement pivots into the required no-pivot exercises.

## Stage 6: connect accuracy, performance, and solver lifetime

Follow source-guide lesson H. Reuse completed factors to compute a correction
from `r=b-A*x`, solve `A*delta=r`, and update `x += delta` in a separate learning
driver. Measure the original-system backward error before and after; refinement
may stagnate and does not guarantee recovery from an unstable factorization.

For a larger experiment, obtain valid supplied plans first. The existing
production-library benchmark harness supplies comparison workflows, not plans
for this teaching solver. Measure analysis, factorization, and solve separately.
Compare one worker with several workers using identical matrices, ordering,
blocks, tolerances, and single-threaded BLAS. Include broad and chain-like DAGs.

- [ ] Explain fill ratio, padding overhead, useful dense work, and map lookup cost.
- [ ] Explain work, critical path, load imbalance, and scheduler overhead.
- [ ] Distinguish reusing factors for a new RHS from reusing analysis for changed
  values. Reusing analysis requires compatible sparsity; pivot-policy constraints
  and a library's reuse options still matter.
- [ ] Reproduce a library solve using the existing baseline harness and record
  commits, options, ordering, pivot policy, thread settings, and error measures.

Exit evidence: a factor/solve lifetime diagram, an accuracy table, a timing table,
and an explanation of one observed bottleneck. Advanced MPI/GPU runtimes and
low-rank compression can follow once the CPU full-rank path is understood.

## Checkpoints and completion

Enable the acceptance targets using the configuration in the
[assignment README](README.md#configure-and-run-acceptance-tests). The supplied
tests cover the integrated drivers; add the direct kernel checks listed above.
An integration test may still reach another TODO while you are partway through
a stage. Completing compilation alone does not establish numerical correctness.

At each stage's checkpoint, retain passing checks from earlier stages, record
factor/solution errors, and complete that stage's report. Test all three exported
index/offset combinations and keep analysis, factorization, and solve timing
separate. On remote servers, submit every build, test, and benchmark through
Slurm; CMake configuration may run on a login node.

## Printable notes and Antigravity IDE

Build this guide or the foundations chapter from Markdown using Pandoc and latexmk:

```bash
make -C docs/solver DOC=README pdf
make -C docs/solver pdf
```

The PDFs are `docs/pdf/README.pdf` and `docs/pdf/foundations.pdf`. Generated LaTeX, SyncTeX, logs,
and other build files stay in `docs/obj/`. Both directories are Git-ignored.
Edit the corresponding Markdown file; the generated `.tex` is replaced when its
source or layout inputs change. Layout inputs are `pdf-header.tex` and
`pdf-layout.lua` in this directory.

In Antigravity IDE with LaTeX Workshop installed:

1. Open the repository root as the workspace. Run **Tasks: Run Task**, then
   **Solver: Build README PDF (Pandoc)** or **Solver: Build foundations PDF (Pandoc)**.
   This works while editing Markdown.
2. Open `docs/obj/README.tex` or `docs/obj/foundations.tex`. Its recipe comment selects
   **Solver notes (Pandoc + latexmk)** for **LaTeX Workshop: Build LaTeX project**.
3. Use **LaTeX Workshop: View LaTeX PDF file** for the in-editor preview. It reads
   the identical build PDF in `docs/obj/`, where SyncTeX can map back to the
   generated `.tex`. The recipe also publishes the final copy to `docs/pdf/`.

After changing Markdown, rerun the task or build from the generated `.tex` tab.
The Pandoc step regenerates LaTeX before compilation. PDF/source synchronization
targets generated LaTeX, not Markdown. The ordinary latexmk recipe remains
available for other LaTeX documents. Recipe setup follows the
[LaTeX Workshop compilation documentation](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile).

The goal is to understand the main CPU sparse-direct-solver pipeline and navigate
the relevant source with a concrete model. GPU runtimes, distributed algorithms,
and low-rank compression remain further study. This is an independent educational
solver with simplified storage and scheduling; the source-reading lessons explain
its relationship to each library.

Shared contracts are in `include/solver/Lu.h`. Public entry points are available
through `SpCraft.h`. Template definitions belong in the corresponding `.cpp`
files; preserve their explicit instantiations. The scaffold exports `double`
with signed `(IT,OT)` widths `(32,32)`, `(32,64)`, and `(64,64)`.

## Mathematical scope

Required implementations use no row pivoting. Dense LU uses A=L*U in original
coordinates; the sparse stages add a supplied permutation and structure.
Use SPD or strictly diagonally
dominant matrices. With `order[new] = old`, define the **symmetric permutation**

```
B(i,j) = A(order[i], order[j]),     B = L U.
rhs[i] = b[order[i]],              x[order[i]] = z[i],     L U z = rhs.
```

An empty permutation means identity. Symmetric permutation preserves the stated
input classes; a column-only permutation does not provide that guarantee for
no-pivot elimination. Both factors are triangular in ordered coordinates. A
rejected pivot is a reported failure, never a reason to silently add epsilon.

The instructor supplies the scalar fill pattern, supernode partition, and
complete contributor lists. `tests/solver/LuFixture.h` supplies small hand-built
plans, including genuine fill and a width-two panel. The parallel stage additionally has a
two-branch join case. General ordering, symbolic analysis, dynamic pivoting,
and creation of plans for large benchmark inputs are outside these assignments.
The instructor must supply such plans before assigning larger performance runs;
this scaffold intentionally does not implement a plan generator.

## Configure and run acceptance tests

Assignment tests are off by default because the numerical bodies are unfinished.
Enable them explicitly; they should fail with the relevant TODO until implemented.
The rest of the project's default test selection is unchanged.

```bash
cmake -S . -B build/solver-assignments \
  -DSPCRAFT_BUILD_DOWNSTREAM=ON -DBUILD_TESTING=ON \
  -DSPCRAFT_BUILD_EXAMPLES=OFF -DSPCRAFT_BUILD_BENCHMARKS=OFF \
  -DSPCRAFT_BUILD_SOLVER_ASSIGNMENTS=ON

# Start with dense, then scalar, supernodal, and mt.
cmake --build build/solver-assignments --target test_solver_dense --parallel 4
ctest --test-dir build/solver-assignments -L solver-dense --output-on-failure
```

Use the repository's normal dependency setup and OpenMP configuration. On remote
servers, run builds, acceptance tests, and benchmarks through Slurm compute-node
jobs; CMake configuration may run on a login node. A serial build must still work
with OpenMP disabled.

## Required evidence

- Verify `P^T A P = L U`, including a case with fill and a nonidentity permutation.
- Form `b = A*x_true`; report forward error and normwise backward error
  `||b-A*x||inf / (||A||inf*||x||inf + ||b||inf)` against the original matrix.
  Handle a zero denominator explicitly. Use `double`; explain tolerances.
- Show reusable solves, input preservation, empty input, and rejected pivots.
  Add tests for the cases described in each stage's instructions. The supplied
  tests are a starting acceptance set, not a complete grader.
- Separate analysis, numerical factorization, and triangular-solve timing. State
  what each timer includes. Use the same matrix, ordering, and plan when comparing
  implementations. Small dense reconstruction is only a testing aid, not the
  factorization implementation.
- Write the dense LU stage directly using elimination and substitution. Do not
  call Eigen, SuperLU, or PaStiX to implement an assignment. The later sparse
  stages must use their prescribed sparse/block algorithms; converting the
  whole sparse matrix to dense LU is outside their contract. Library routines
  may serve as independent references.

Suggested grading: correctness 50%, explanation and invariants 25%, performance
investigation 15%, and code clarity 10%. Reward an accurate explanation of a
slowdown; do not require a fixed speedup on an arbitrary shared machine.

[Reference-source notes](../direct-solver-references.md) and
[measured CPU baselines](../../benchmarks/direct/BASELINE_LAIR_20261022.md)
provide context. Their pivot policies and orderings differ from each other and
from this restricted teaching solver.
