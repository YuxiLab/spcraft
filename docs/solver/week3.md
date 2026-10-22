# Week 3: dependency-driven multithreaded LU

Goal: schedule the Week 2 numerical kernels concurrently while preserving their
data dependencies and storage lifetimes. The numerical representation stays the
same, and the serial Week 2 `Solve` is reused.

Implement W3.1–W3.2 in `src/solver/mt/Lu.cpp` following
`include/solver/mt/Lu.h`. Do not replace the Week 2 kernels with another solver.

## Tasks

1. **W3.1 `BuildTaskGraph`**: invert the supplied contributor lists into a CSR
   successor graph and compute indegrees. Reject malformed IDs, self/forward
   edges, and duplicates. The earlier-ID condition makes the supplied graph a
   DAG. The graph includes every numerical read dependency; etree parent edges
   alone are not a substitute for this contract.
2. **W3.2 `Factorize`**: allocate the complete output panel vector before launching
   workers. Execute only panels whose contributors have successfully completed.
   Each active panel owns a workspace and is the sole writer of its output slot.
   Invoke the Week 2 assembly/update/factor sequence. Publish completion only
   after the numerical data is visible, then make successors eligible.

A synchronized ready queue is sufficient. Use the existing OpenMP wrapper for
the worker region; protect queue operations and dependency counters as needed.
Alternatively use OpenMP tasks with explicit dependencies and an appropriate
serial fallback; extend the wrapper narrowly if that route needs task pragmas.
Do not use `parallel for` over all panel IDs or merely protect the entire
factorization with one critical section.

Handle failure as part of scheduling. Do not release successors of a rejected
pivot. Join outstanding workers and return unsuccessful factors without a
deadlock; choose the smallest pivot column among failures actually reported by
workers. Capture worker exceptions and rethrow after joining. Do not let an
exception escape an OpenMP region. Do not resize shared vectors while workers
hold references to their elements.

Keep BLAS single-threaded. Honor a positive requested thread count, and retain
correct serial behavior when OpenMP is disabled. Avoid changing process-wide
thread settings unexpectedly.

## Checks and submission

Run `test_solver_mt` / CTest label `solver-week3`. Check thread counts 1,2,4,8,
the provided fork/join graph, invalid edges, and pivot failure without deadlock.
Add repeated runs, multiple independent subtrees, an empty graph/matrix, and a
numerical comparison with Week 2. Supplied numerical tests establish results;
you must also explain why no consumer can observe incomplete panel data.

For performance, use instructor-supplied plans with both broad concurrency and
a long dependency chain. Time analysis, factorization, and solve separately;
measure factorization speedup against the one-thread version of the same code.
Use identical ordering, blocks, BLAS settings, and inputs across thread counts.
On cluster systems, all builds and runs must go through Slurm compute nodes.

Submit code and a short report with:

- The task DAG and the eligibility/completion invariants.
- Which thread owns each buffer and which operations synchronize publication.
- The behavior when the queue is empty but other tasks are still active.
- A speedup table or plot with several repetitions and explanations of overhead,
  imbalance, and critical-path limits. A slowdown is a valid measured result.
- A description of failure propagation and your deadlock checks.

Read SuperLU_MT's `pxgstrf_scheduler.c` and PaStiX's dynamic worker in
`sopalin/sequential_zgetrf.c`. Compare your conservative ready-only schedule with
SuperLU_MT's pipelining and PaStiX's queues/work stealing. Neither advanced feature
is required here; parallel triangular solves, MPI, and GPUs are also outside scope.
