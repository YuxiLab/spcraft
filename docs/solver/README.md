# Three-week sparse LU assignments

Build one solver in three stages: scalar sparse elimination, serial supernodal
factorization, and shared-memory panel scheduling. The C++ assignment directories
contain interfaces and TODO stubs, **not reference implementations**. Calling
an unfinished function throws `std::logic_error` with its assignment ID.

| Week | Implementation | Instructions | Acceptance target |
|---|---|---|---|
| 1 | `include/solver/scalar/`, `src/solver/scalar/` | [Scalar LU](week1.md) | `test_solver_scalar` |
| 2 | `include/solver/supernodal/`, `src/solver/supernodal/` | [Supernodal LU](week2.md) | `test_solver_supernodal` |
| 3 | `include/solver/mt/`, `src/solver/mt/` | [Parallel LU](week3.md) | `test_solver_mt` |

Shared contracts are in `include/solver/Lu.h`. Public entry points are available
through `SpCraft.h`. Template definitions belong in the corresponding `.cpp`
files; preserve their explicit instantiations. The scaffold exports `double`
with signed `(IT,OT)` widths `(32,32)`, `(32,64)`, and `(64,64)`.

## Mathematical scope

Required implementations use no row pivoting. Use SPD or strictly diagonally
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
plans, including genuine fill and a width-two panel. Week 3 additionally has a
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

# Substitute supernodal/week2 or mt/week3 in subsequent weeks.
cmake --build build/solver-assignments --target test_solver_scalar --parallel 4
ctest --test-dir build/solver-assignments -L solver-week1 --output-on-failure
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
  Add tests for the cases described in each week's instructions. The supplied
  tests are a starting acceptance set, not a complete grader.
- Separate analysis, numerical factorization, and triangular-solve timing. State
  what each timer includes. Use the same matrix, ordering, and plan when comparing
  implementations. Small dense reconstruction is only a testing aid, not the
  factorization implementation.
- Never use Eigen, SuperLU, PaStiX, or a dense whole-matrix factorization to
  implement an assignment body. They may serve as independent references.

Suggested grading: correctness 50%, explanation and invariants 25%, performance
investigation 15%, and code clarity 10%. Reward an accurate explanation of a
slowdown; do not require a fixed speedup on an arbitrary shared machine.

[Reference-source notes](../direct-solver-references.md) and
[measured CPU baselines](../../benchmarks/direct/BASELINE_LAIR_20261022.md)
provide context. Their pivot policies and orderings differ from each other and
from this restricted teaching solver.
