# SuperLU_MT and PaStiX as SpCraft references

The initial source checkouts and CPU baselines are described in
[the baseline harness](../benchmarks/direct/README.md). This note records the
design differences that matter when choosing SpCraft's first implementation.

| Capability | SuperLU_MT | PaStiX |
| --- | --- | --- |
| Numerical factorizations | General sparse LU with threshold partial row pivoting | LU with static pivoting; Cholesky and symmetric/Hermitian variants |
| Shared-memory execution | Pthreads or OpenMP; factorization is parallel, ordering and triangular solves are sequential | Internal sequential, static, and dynamic schedulers; parallel factorization and solve routines |
| Larger execution targets | Shared-memory CPU solver | Optional MPI, StarPU, PaRSEC, and GPU support; availability varies by backend |
| Accuracy support | Equilibration, iterative refinement, condition estimates, forward/backward error estimates, pivot-growth diagnostics in the expert driver | Iterative refinement, including Krylov methods; static-pivot replacement statistics |
| Other major capabilities | Reuse of factors and selected analysis/pivot information | Schur-complement interfaces, incomplete factorization options, block low-rank compression |
| Precision | Single/double, real/complex | Single/double, real/complex |

Sources: [SuperLU_MT README](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/README),
[expert driver](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgssvx.c),
[PaStiX feature matrix](https://solverstack.gitlabpages.inria.fr/pastix/index.html),
and [PaStiX parameters](https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__api.html).

## What to read in SuperLU_MT

The `p` prefix identifies the parallel interface, and `d` identifies double
precision. Read the following files in the downloaded `superlu_mt/SRC/` tree:

1. `pdgssv.c` and `pdgssvx.c`: simple and expert driver contracts, permutations,
   factor reuse, solve, and diagnostics. The basic factorization is
   `Pr * A * Pc = L * U`.
2. `pdgstrf_init.c` and `sp_colorder.c`: column permutation, elimination-tree
   preprocessing, structural information, and options needed by factorization.
3. `pdgstrf_thread.c`: the worker loop, per-thread sparse accumulators and DFS
   workspaces, relaxed-supernode handling, and panel factorization.
4. `pxgstrf_scheduler.c`: a shared panel queue, parent preference, ready-panel
   scheduling, and optional pipelining while descendants are busy.
5. `pdgstrf_panel_dfs.c` and `pdgstrf_panel_bmod.c`: discover contributing
   supernodes, gather their updates into a panel, and wait for unfinished
   contributors when using pipelining.
6. `pdgstrf_pivotL.c`: threshold partial pivoting and the interaction between
   numerical row selection and factor storage.
7. `pdmemory.c`, `supermatrix.h`, and `slu_mt_util.h`: storage ownership,
   parallel allocation, supernode descriptors, and synchronization state.
8. `dgstrs.c`: forward/backward solves against the completed factors.

The key computation is a left-looking supernode-to-panel update. Supernodes
are used during numerical factorization, rather than packed only after a
scalar sparse factorization. Its asynchronous scheduler is closely coupled
to the factor storage and numerical dependency tracking.

Do not infer incomplete-LU support from the `drop_tol` option: this checkout's
`pdgstrf_init.c` explicitly marks that parameter as not implemented. Likewise,
the multithreaded factorization does not imply a parallel triangular solve.

## What to read in PaStiX

Read the following files in the downloaded `pastix/` tree:

1. `include/pastix.h` and `example/simple.c`: solver lifetime and the explicit
   `analyze -> numfact -> solve -> refine` sequence.
2. `blend/pastix_task_analyze.c`: ordering, symbolic factorization, block
   reordering, mapping, and generation of the numerical solver structures.
3. `blend/solver.h`: `SolverCblk`, `SolverBlok`, task metadata, dependency
   counters, and work-stealing support.
4. `sopalin/pastix_task_sopalin.c` and `sopalin/sequential_zgetrf.c`: conversion
   into numerical storage and the LU factorization scheduling paths. The
   precision-generation machinery derives real and other precision variants
   from the source templates.
5. `kernels/core_zgetrfsp.c`: diagonal-block LU with static pivot replacement.
   Small diagonal entries are replaced according to a threshold; this is
   different from searching for and interchanging pivot rows as in SuperLU_MT.
6. `sopalin/sequential_ztrsm.c`, `sopalin/pastix_task_solve.c`, and
   `refinement/pastix_task_refine.c`: solve scheduling and iterative refinement.

PaStiX is especially useful as a reference for separating symbolic structure,
numerical coefficient storage, dense block kernels, and execution scheduling.
Its [analysis documentation](https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__analyze.html)
explains the preprocessing stages and the generated block solver structure.

## Suggested first SpCraft scope

Start with double-precision, square CSC matrices, one CPU node, LU, a reusable
analysis/factor/solve interface, and a serial triangular solve. Support
singularity reporting and validate residuals before performance work. Keep
index and offset types explicit and retain separate column and row permutation
semantics.

Use SuperLU_MT as the numerical reference if the goal is general unsymmetric
LU with partial pivoting. Use PaStiX as the architectural reference for explicit
phase boundaries, block descriptors, and ownership of scheduling state.
Dynamic partial pivoting requires numerical structure and row-ownership
handling; a fixed symbolic allocation designed for static pivoting cannot
simply be substituted without establishing the required structural bounds.

Build a correct serial supernodal factorization first. Then add conservative
ready-panel execution with one writer per active panel, stable factor storage,
and explicit publication of completed data. Prove that the dependency and
pivot-ownership rules make concurrent panels safe before enabling them. Defer
pipelining, work stealing, parallel triangular solves, distributed execution,
GPU kernels, and low-rank compression until that baseline is correct and
measured.
