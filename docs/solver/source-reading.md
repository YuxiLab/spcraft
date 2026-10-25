# From your LU kernels to SuperLU_MT and PaStiX

Read this alongside the [implementation sequence](README.md), one
lesson at a time. For each routine, identify its inputs, outputs, coordinate
systems, storage owner, and completion condition before following inner loops.
Write down one shared idea and one additional responsibility absent from your code.

The local reference trees are ignored downloads, not a dependency of these
exercises. The reviewed checkouts are:

- SuperLU_MT: `refernce/sparse-direct/superlu_mt`, commit
  `f2e50bfcbb7b20ef304c84cf92c038c283019863`.
- PaStiX: `refernce/sparse-direct/pastix`, commit
  `ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2`.

Check your checkout's commit before relying on field names. SuperLU_MT links
below are pinned. PaStiX source links are pinned too; its generated handbook
describes the documentation build's version, which may differ from this checkout.

## Concept map

| Your learning model | SuperLU_MT connection | PaStiX connection |
|---|---|---|
| Dense packed LU and substitution | Multipliers, triangular solves, and outer-product arithmetic beneath panel kernels | The same block algebra beneath diagonal GETRF/TRSM kernels |
| `LuSymbolic` and `order` | Column preprocessing plus structure discovered during factorization; additional row-pivot maps | Analysis: ordering, symbolic structure, reordering, mapping |
| Scalar accumulator and markers | SPA, DFS markers, segments, and temporary dense buffers | Useful preparation for sparse-to-block coefficient assembly |
| `Panel` with J/R/C | Supernodes and supernode-to-panel updates; numerical panels have a separate role | Column blocks, off-diagonal blocks, coefficient buffers |
| Diagonal LU, two block solves, updates | Numerical kernels within panel/column factorization | GETRF, TRSM, and GEMM kernels |
| Contributor DAG and private workspace | Shared panel scheduling, private work arrays, pipelining | Contribution counts, local queues, dynamic scheduling |
| Rejected pivot | Threshold partial row pivoting | Static pivot replacement in the studied LU kernels |
| Reusable factors and original residual | Factor reuse, triangular solves, expert-driver diagnostics | Analyze/numfact/solve/refine lifecycle |

These are conceptual correspondences, not interchangeable data structures.
The scaffold implements one complete cross per planned interval with ready-only
execution. It does not reproduce either production solver's storage or runtime.

## Lesson A: drivers and triangular solves

First complete the dense stage to distinguish the basic algebra from sparse
storage. Its row-by-row solves use the same equations as sparse column updates
but visit coefficients in a different order; the storage traversal changes.

Read after W1.3/W1.4. Start with the purpose and permutation contract of
[SuperLU_MT `pdgssv.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgssv.c),
then locate the forward/backward work in
[`dgstrs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/dgstrs.c).
First follow one RHS and the non-transposed solve. Leave other paths for later.

Find where row and column permutations enter, how factor storage differs from
plain CSC, and which work is numerical substitution. Draw the lifetime
`input -> factorization -> factors -> solve -> another RHS -> free`.
The [upstream README](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/README)
states that factorization is multithreaded while ordering and triangular solves
are sequential in this library. Your serial solve after parallel factorization
has the same broad division of work.

Explain: why does solving a second RHS not require new fill analysis? Which
permutations belong to a driver, and which belong inside a triangular kernel?

## Lesson B: scalar work, SPA, and structural discovery

Read after W1.1/W1.2/W1.5. Follow
[`pdgstrf_column_bmod.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_column_bmod.c)
for numerical updates and
[`pdgstrf_column_dfs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_column_dfs.c)
for the work needed to discover contributing structure. Use the local routine
comments to identify `dense`, `segrep`, `repfnz`, markers, and row subscripts.

Your planned U column supplies the earlier pivots to visit. SuperLU_MT carries
additional segment and reach information so useful work can be found and grouped.
Trace one coefficient from the original A, through an accumulator update, into
the factor storage. Identify which arrays contain values and which describe
where values can occur. A structural marker is not a floating-point zero test.

Explain: what does the supplied plan save you from implementing? Why must a
contributor be applied before extracting the U coefficient that depends on it?
Record where row pivoting will complicate the mapping, but study its policy in G.

The graph vocabulary matters here. Reach finds structural vertices that can
affect a column; DFS explores that reachable structure and produces an order
usable for elimination. Segments group U entries associated with a source
supernode. Pruning reduces future graph searches using already established
structure; it is not dropping small coefficients from an incomplete LU.
Identify `xprune` and `ispruned`, then explain the invariant that permits a
particular search to stop early. You do not need to reproduce this optimization.

Inspect
[`supermatrix.h`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/supermatrix.h),
[`slu_mt_util.h`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/slu_mt_util.h),
and
[`pdmemory.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdmemory.c).
Draw one column in `SCPformat` and `NCPformat`: numerical-column bounds and
compressed row-index bounds need not be the same arrays. Column begin/end
positions allow storage that is not one simple contiguous CSC sequence.
Compare this flexibility and its allocation/synchronization costs with your
fixed owning buffers. Find `xsup`, `supno`, `lsub`, and `lusup` in the internal
factor structures and connect them to column intervals, maps, and coefficients.

## Lesson C: supernodes, panels, and dense arithmetic

Read after W2.1/W2.2/W2.3. Compare
[`pdgstrf_panel_dfs.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_panel_dfs.c),
[`pdgstrf_panel_bmod.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_panel_bmod.c),
and the inner column work in
[`pdgstrf_thread.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_thread.c).

Distinguish a structural supernode, a relaxed supernode, a numerical panel, and
an updating supernode segment. In your code `Panel` combines a planned interval
and its cross; in SuperLU_MT the panel is a batch of columns being processed,
and multiple earlier supernodes can update it. A panel width is therefore not
automatically the width of a structural supernode.

Identify gather/scatter work, triangular solves, and matrix products. Select
one update and label its dimensions, row maps, and leading dimensions. Explain
the reason for separate small and larger block paths without copying their tuning.

Explain: how do scalar outer products become block updates? What is lost if you
factor scalar CSC first and only pack the completed answer into dense blocks?
Why can storing some zeros help, and when might the extra work hurt?

## Lesson D: PaStiX blocks and solver lifetime

Read after the complete serial block solve. Start with
[`example/simple.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/example/simple.c)
and the [public task API](https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__users.html).
Locate analyze, numerical factorization, solve, and refinement as separate stages.

Next inspect
[`blend/solver.h`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/blend/solver.h).
Map `SolverCblk` to a column interval and its block list, and `SolverBlok` to
an off-diagonal block's rows and facing column block. Observe that `lcolnum`
and `lrownum` are inclusive bounds in these descriptors; your `last` is exclusive.
Distinguish structural descriptors, allocated numerical buffers, and runtime
counters. The [descriptor documentation](https://solverstack.gitlabpages.inria.fr/pastix/solver_8h.html)
helps locate fields but does not replace checking a pinned source definition.

In
[`sopalin/pastix_task_sopalin.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/sopalin/pastix_task_sopalin.c),
follow `pastix_task_numfact` through `pastix_subtask_spm2bcsc`,
`pastix_subtask_bcsc2ctab`, and `pastix_subtask_sopalin`. Separate conversion to
block CSC, filling the solver's coefficient buffers, and executing numerical
factorization. The symbolic plan determines the blocks, while current input
values populate them. Your `AssemblePanel` performs a smaller related gathering
job inside each numerical task rather than reproducing this complete lifecycle.

Read the full-rank, CPU path in
[`kernels/core_zgetrfsp.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/kernels/core_zgetrfsp.c).
Locate diagonal factorization and off-diagonal work, then follow the TRSM/GEMM
calls into their kernels as needed. The `z` template is complex double; generated
`d` routines represent real double. Follow transpose flags and coefficient layout:
this checkout explicitly transposes the factored diagonal into its U coefficient
buffer. Do not assume it stores your literal A12/L21/U12 cross unchanged.

Explain: which parts of your `Panel` are numerical coefficients, which are maps,
and which PaStiX metadata additionally describes execution or distribution?

## Lesson E: workers, publication, and different update ownership

Read after W3.1/W3.2. Compare your scheduler with
[`pxgstrf_scheduler.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pxgstrf_scheduler.c)
and the worker loop in `pdgstrf_thread.c`. Locate parent preference, ready
panels, and panels eligible for pipelining. Follow how the panel update kernel
handles completed versus busy contributors. Your ready-only algorithm is an
intentional first step before reasoning about partial progress and waits.

For PaStiX, read `sequential_zgetrf`, `thread_zgetrf_static`, and
`thread_zgetrf_dynamic` in
[`sopalin/sequential_zgetrf.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/sopalin/sequential_zgetrf.c).
Despite the filename, it contains several scheduler paths. Locate local ready
queues, contribution counters, and stealing. Then follow one update into
`cpucblk_zgetrfsp1d` and its update kernels.

Your destination task gathers completed sources and writes only its own cross:
left-looking ownership. A right-looking path pushes a factored source's updates
into later blocks. Determine the selected library path before assigning a
direction. A right-looking update needs a protocol for concurrent contributions
to the same destination; copying your single-writer assumption is insufficient.
Scheduling policy and update direction are separate choices.

Explain: what makes a task eligible, who writes a destination, and which operation
publishes data? Draw an event trace for `0->2<-1`. Show why a bare atomic decrement
without an appropriate data-visibility protocol is not a complete proof.

## Lesson F: ordering, symbolic analysis, and the elimination tree

Read during Stage 4. Inspect
[`sp_colorder.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/sp_colorder.c)
and
[`pdgstrf_init.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_init.c),
then connect them to the DFS and relaxed-supernode work already encountered.
For PaStiX, follow
[`blend/pastix_task_analyze.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/blend/pastix_task_analyze.c)
and the [analysis documentation](https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__analyze.html).
Separate ordering, symbolic structure, block reordering, mapping, and generation
of numerical execution structures.

On a tiny symmetric-pattern example, derive fill sets and structural ancestry
by hand. For SuperLU_MT's unsymmetric path, inspect the documented tree of
`A^T*A`; do not interpret every library etree as the symmetric graph of A itself.
Tree edges summarize reach/ancestry. Your full contributor list directly names
every factor panel the chosen numerical kernel reads.

Explain: why is choosing an ordering different from discovering fill? Which
symbolic information can survive new coefficient values, and what assumptions
would dynamic row pivoting add? Why does a relaxed supernode store extra zeros?

## Lesson G: pivoting and numerical stability

Read during Stage 5. Inspect
[`pdgstrf_pivotL.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgstrf_pivotL.c).
Find the column maximum, the threshold comparison, optional reuse of an existing
pivot row, diagonal preference, updates to `perm_r`/`inv_perm_r`, and row movement
within supernodal storage. Track actual mapping directions rather than inferring
them from variable names.

In PaStiX's `core_zgetrfsp.c`, start with `core_zgetf2sp`: identify the comparison
with `criterion`, replacement of small diagonal values, and the pivot counter.
Follow where the numerical driver sets the criterion; it is not automatically
the same quantity as the scaffold's absolute rejection tolerance.

| Policy | Numerical action | Question to answer |
|---|---|---|
| Assignment | Reject a bad diagonal before division | How is failure propagated without exposing unusable factors? |
| SuperLU_MT | Choose a pivot row by a relative threshold policy | Which row maps and previously stored values must remain consistent? |
| Studied PaStiX kernel | Replace a small diagonal under its static policy | How is the modification counted and how accurate is the original-system solve? |

Use the two Stage-5 matrices. Report actual multipliers and original residuals.
Static pivoting is not synonymous with simply omitting row interchanges; in
these kernels it can modify a small pivot. Adding row selection to your fixed
cross format requires a new structural and concurrent-ownership argument.

## Lesson H: accuracy, reuse, and performance

Read during Stage 6. Follow
[`pdgssvx.c`](https://github.com/xiaoyeli/superlu_mt/blob/f2e50bfcbb7b20ef304c84cf92c038c283019863/SRC/pdgssvx.c)
for equilibration, reuse options, condition/error diagnostics, and refinement.
Read PaStiX's
[`refinement/pastix_task_refine.c`](https://gitlab.inria.fr/solverstack/pastix/-/blob/ac716d6259eb014cd8874a3f0f4c77f6d5d2f6c2/refinement/pastix_task_refine.c)
and [refinement documentation](https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__refine.html).
Identify which correction method is selected rather than treating every
refinement implementation as the simple stationary correction experiment.

Explain the diagnostic vocabulary before interpreting outputs. Equilibration
rescales rows/columns and therefore adds transformations around the solve.
A condition estimate describes sensitivity to perturbations; a residual alone
does not supply that information. Pivot growth concerns enlargement of entries
during elimination and helps assess numerical behavior. Iterative refinement
uses the original-system residual and factor-based correction solves, with
convergence depending on the problem, factors, and arithmetic. Record which
diagnostics a particular driver actually computes and the norm/convention used.

Use [the existing benchmark harness](../../benchmarks/direct/README.md) to study
how source versions, inputs, error checks, BLAS threads, and timings are recorded.
Different orderings, pivot policies, fill, and block formats confound a direct
scheduler comparison. Factorization speedup within one fixed solver configuration
answers a narrower and more interpretable question.

Exit assessment: using a small matrix, explain the route from input CSC through
ordering, symbolic structure, numerical blocks, dependency scheduling, pivot
policy, solve permutations, and residual/refinement. Point to a routine for each
stage in both libraries and state where your learning solver simplifies it.
