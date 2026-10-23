# Student implementation order

Follow this checklist from top to bottom. Assignment IDs identify the original
tasks; they do not have to match implementation order. Begin with the two small
triangular solves so you can verify them independently before implementing LU.
Each function body contains numbered **Part** comments describing its stages.
Replace its final TODO exception with your implementation as you complete it.

Read the [shared contract](README.md#mathematical-scope) and
[`include/solver/Lu.h`](../../include/solver/Lu.h) first. The instructor supplies
ordering and symbolic plans; you implement numerical kernels and scheduling.
Keep the public signatures, owning matrix/vector types, and Boost.Preprocessor
instantiations. The compiled combinations use `double` with `(IT,OT)` widths
`(32,32)`, `(32,64)`, and `(64,64)`.

## Week 1: scalar sparse LU

Work in [`src/solver/scalar/Lu.cpp`](../../src/solver/scalar/Lu.cpp), following
[`include/solver/scalar/Lu.h`](../../include/solver/scalar/Lu.h) and
[Week 1 instructions](week1.md).

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 1 | `scalar::ForwardSolve` / W1.3 | CSC column traversal and unit-lower structure | Hand-built L: verify `L*y=rhs`, explicit diagonal handling, unchanged RHS, and empty input. |
| 2 | `scalar::BackwardSolve` / W1.4 | CSC column traversal and upper structure | Hand-built U: verify `U*x=rhs`, reverse traversal, and rejection of zero/nonfinite diagonals. |
| 3 | `scalar::FactorColumn` / W1.1 | Ordered coordinates, supplied L/U patterns, accumulator | With manually allocated patterns and scratch, factor columns sequentially; verify a column with fill, stored cancellation zeros, and scratch cleanup on success/failure. |
| 4 | `scalar::Factorize` / W1.2 | `FactorColumn` | Validate/allocate once, run the column loop, and check `P^T*A*P=L*U`, malformed plans, empty input, and rejected pivots. |
| 5 | `scalar::Solve` / W1.5 | `ForwardSolve`, `BackwardSolve`, successful factors | Check a nonidentity permutation, original-coordinate residual, unchanged inputs, and reuse for a second RHS. |

- [ ] Implement and directly check `ForwardSolve`.
- [ ] Implement and directly check `BackwardSolve`.
- [ ] Implement `FactorColumn` using preallocated factors and reusable scratch.
- [ ] Implement `Factorize` and verify factor reconstruction.
- [ ] Implement `Solve` and pass `test_solver_scalar` / `solver-week1`.

The driver allocates the exact symbolic patterns and constructs `inverse_order`
once. `FactorColumn` performs numerical work on one column; it does not discover
fill or allocate new factor entries. During elimination, the accumulator holds
the scattered ordered column minus contributions already applied. Record each
touched accumulator location once, including locations reached through fill,
and clear it before the workspace is reused.

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

## Week 2: serial supernodal LU

Work in [`src/solver/supernodal/Lu.cpp`](../../src/solver/supernodal/Lu.cpp), using
the layouts in [`include/solver/supernodal/Lu.h`](../../include/solver/supernodal/Lu.h)
and [Week 2 instructions](week2.md). Finish Week 1 first so its factors and solves
can serve as an independent numerical comparison.

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 6 | `supernodal::AssemblePanel` / W2.1 | Permutation mapping and column-major blocks | Compare A11/A21/A12 with a small ordered matrix; verify zero initialization and reuse of workspace for different panel sizes. |
| 7 | `supernodal::UpdatePanel` / W2.2 | `AssemblePanel`; hand-built completed contributor factors | Verify all three update products against a dense calculation, including multiple contributors, missing couplings, and distinct row/column maps. |
| 8 | `supernodal::FactorPanel` / W2.3 | Updated cross and dense no-pivot elimination | Check `A11=L11*U11`, `A21=L21*U11`, and `A12=L11*U12`; reject a later local pivot using its global column. |
| 9 | `supernodal::Factorize` / W2.4 | All three panel kernels | Validate the partition, run the serial pipeline, and reconstruct the factors for width-one and wider panels. |
| 10 | `supernodal::Solve` / W2.5 | Successful block factors and triangular substitution | Match the scalar solution with a nonidentity permutation, implicit L diagonal, empty off-diagonal blocks, and repeated RHS solves. |

- [ ] Implement `AssemblePanel` and check each gathered block.
- [ ] Implement `UpdatePanel` with manually constructed source panels.
- [ ] Implement `FactorPanel` and check the three factor equations.
- [ ] Implement `Factorize` and compare with scalar LU.
- [ ] Implement `Solve` and pass `test_solver_supernodal` / `solver-week2`.

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

## Week 3: multithreaded factorization

Work in [`src/solver/mt/Lu.cpp`](../../src/solver/mt/Lu.cpp), following
[`include/solver/mt/Lu.h`](../../include/solver/mt/Lu.h) and
[Week 3 instructions](week3.md). Finish and verify the serial block kernels first.

| Order | Function / assignment | Prerequisites | Check before moving on |
|---|---|---|---|
| 11 | `mt::BuildTaskGraph` / W3.1 | Complete contributor lists and CSR prefix sums | Hand-check successor ranges/indegrees for a fork/join graph; test empty input, duplicates, and invalid/self/forward edges. |
| 12 | `mt::Factorize` / W3.2 | `BuildTaskGraph` and Week 2 numerical kernels | Match serial factors/solutions, then test 1/2/4/8 threads, repeated runs, independent branches, and pivot/exception paths without deadlock. |

- [ ] Implement `BuildTaskGraph` and check its CSR arrays directly.
- [ ] Implement `Factorize` first with one worker using the ready queue.
- [ ] Enable multiple workers with private workspaces and synchronized publication.
- [ ] Add stop/drain handling for numerical failures and worker exceptions.
- [ ] Pass `test_solver_mt` / `solver-week3` and run the additional concurrency checks.

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

## Checkpoints and completion

Enable the acceptance targets using the configuration in the
[assignment README](README.md#configure-and-run-acceptance-tests). The supplied
tests cover the integrated drivers; add the direct kernel checks listed above.
An integration test may still reach another TODO while you are partway through
a week. Completing compilation alone does not establish numerical correctness.

At each week's checkpoint, retain passing checks from earlier weeks, record
factor/solution errors, and complete that week's report. Test all three exported
index/offset combinations and keep analysis, factorization, and solve timing
separate. On remote servers, submit every build, test, and benchmark through
Slurm; CMake configuration may run on a login node.
