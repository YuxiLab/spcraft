# CombBLAS-mapped SpGEMM baseline

`OmpHashSpGEMM<Ring>(A,B)` is the explicit source-mapped DCSC baseline.
Its reference is `LocalSpGEMMHash<SR,NT>(A,B,false,false,true)` at CombBLAS commit
`2381a1f7a028b9f56d8531c0b7b157ae2ab27572`. It returns native sorted tuples.
The native-tuple path in `SpGEMMBenchmark.cpp` calls this baseline.
The result is `CooMatrix<IT,NT,OT>` with a raw `Entry* entries` array, where
`Entry` is exactly `std::tuple<IT,IT,NT>`. Consolidating the tuple owner into COO
does not add conversion or change any hash or semiring operation.

```cpp
#include "SpCraft.h"
using Ring = spcraft::PlusTimesRing<double>;
// A and B are canonical DcscMatrix<int,double>, A.n == B.m.
auto C = spcraft::OmpHashSpGEMM<Ring>(A, B);
// C.entries[p] is std::tuple<int,int,double>: (row,column,value).
```

This is an algorithmic/source mapping with the explicit adaptations below,
not a claim of identical generated machine instructions or allocator calls.
This is the sole maintained SpCraft SpGEMM kernel. The former CSR and general
CSC/DCSC implementations, output selectors and profiler ablations have been
removed. Their historical differences are recorded in
[GeneralKernelMapping.md](GeneralKernelMapping.md); that document does not
describe an available implementation.

## Scope

Inputs are well-formed, immutable, canonical DCSC: sorted unique stored column
IDs, sorted unique row IDs within each nonempty column, and valid offsets.
Compare with OT=IT and equal NT types to match upstream's storage widths.
The mapped lookup requires A.n+1 to fit IT and be at most 2^24, retaining
upstream's float arithmetic within its exact-integer range. Empty operands
return before lookup construction. Work totals and output totals must fit OT.

The source equivalence argument applies where upstream's signed hash products,
signed symbolic capacities, int-sized loop/prefix lengths and unsigned casts
are representable. SpCraft does not reproduce undefined behavior outside that
range. No input conversion is hidden inside the baseline.

## One-to-one stage correspondence

The implementation is
[mtSpGEMM.h](../../include/kernel/mtSpGEMM.h).
Upstream references are
[mtSpGEMM.h](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/mtSpGEMM.h)
and
[dcsc.cpp](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/dcsc.cpp).

| Upstream block | Mapped block | Preserved logic |
|---|---|---|
| mt 474–478: zero operand | 308–312 | Return an empty tuple product with A.m by B.n shape |
| dcsc 1043–1071: ConstructAux | 117–141: lookup constructor | Same float ceil((n+1)/nzc), chunk count, advancing column/chunk boundaries; no dense-column shortcut |
| dcsc 1363–1425: FillColInds | 142–180 | Same integer nzc/nind<4 selection; same pair intersection or auxiliary lookup; same missing-column (0,0) range |
| dcsc 1144–1155: AuxIndex | 168–176 | Same bucket expression and linear std::find |
| mt 490–497: thread discovery | 43–55 | Read actual team size; single writer replaces the upstream race |
| mt 1058–1137: estimateFLOP | 203–234: CombBLASEstimateFLOP | Fresh lookup, zero work counts, per-thread ranges, complete sum of A-range lengths; no height saturation or early stop |
| mt 501: work prefix | 316 | Retain the full work prefix before symbolic, even though it does not drive sorted output |
| mt 24–69: prefixsum | 68–107: CombBLASPrefixSum | Static local prefix, publish per-thread total, barrier, each thread sums preceding totals, second identical static partition adds offset |
| mt 807–934: estimateNNZ_Hash | 237–282: CombBLASEstimateNNZHash | Fresh lookup, per-thread colinds and key vectors; grow/reuse scratch across columns |
| mt 883–905: symbolic sizing/init | 261–264 and 184–191 | Start at 16, double until >=full flop count, initialize active key slots to IT(-1); zero work still gets 16 slots |
| mt 907–930: symbolic traversal/probing | 265–279 | Same B-entry/A-range order; key-hit branch first, empty insertion/count second, stride-one collision advance |
| mt 506–513: output prefix/free | 318–323 | Prefix exact symbolic counts and release work/count arrays before numeric execution |
| mt 516: tuple array | 324 | Allocate exactly output nnz live std::tuple<IT,IT,NT> entries |
| mt 519–547: numeric range scratch | 325–340 | Per-thread colinds initial size nnzA/thread count; grow if needed, materialize ranges before hashing |
| mt 553–572: numeric capacity/init | 341–344 | Minimum 16, doubled to exact column nnz; default-initialized pairs, keys IT(-1) |
| mt 576–604: numeric traversal | 345–364 | Same B-entry/A-entry order; Multiply(A_value,B_value); hit uses Add(product,accumulator); new key assigns product directly |
| mt 607–618: compact/sort | 365–369 | Ascending bucket scan, occupied prefix compaction, row-key sort |
| mt 620–623: emit tuples | 370–372 | Write (row,B.col_id[i],value) into the exact prefix slice |
| mt cleanup/return | Scoped owners and return at 374 | Inputs remain borrowed for clearA=clearB=false; scratch and output have explicit owners |

All expansion loops retain the reference's absent `schedule` clause. The
prefix loops explicitly use static scheduling, as upstream does. An absent
clause has an implementation-defined default; claiming it mandates static
on every OpenMP implementation would be incorrect. Scratch reuse and this
scheduling choice are deliberate exceptions to the general kernel conventions,
authorized by the request for a mapped baseline.

## Arithmetic and hash-state correspondence

For a candidate B column, identical FillColInds ranges imply an identical
ordered stream of A row keys and input-value pairs. Full flop counts yield
the same symbolic capacity. Both tables start with IT(-1), hash with multiplier
107 and mask capacity-1, test equality before emptiness, and advance one slot
on collision. Induction over this key stream establishes matching symbolic
table states and distinct counts in the common representable range.

Equal distinct counts imply equal numeric capacities and output prefixes.
The numeric stream is the same. On a miss, both assign the first product;
on a hit, both call Add(product,accumulator). Thus they perform the same
semiring operation sequence and arguments within each column. The same table
compaction and row ordering then emit the same tuple sequence. Scheduling
can interleave different columns; it does not reorder arithmetic within one
column. Semiring operations must be nonthrowing and thread-safe.

Unused numeric slots are default-initialized, exactly as upstream. They are
never read before the first product is assigned. This deliberately avoids the
former general kernel's Add(identity,first_product), including its signed-zero change.
This reference-specific insertion rule supersedes the general accumulator
initialization convention for this baseline only.

## Explicit adaptations

- Dimension validation and checked element counts precede access/allocation.
  The lookup rejects unsupported float/integer extents instead of inheriting
  upstream overflow or lossy indexing.
- Work and prefix additions check overflow. Worker code records a flag and
  throws after the parallel region; there is no kernel try/catch. The prefix
  partition and per-thread arithmetic order otherwise follow upstream.
- Before symbolic and numeric loops, a serial maximum-capacity check validates
  byte size and vector representability. Workers still double capacity from
  flop or offset differences exactly as upstream; no capacity array or capped
  work estimate replaces that calculation.
- Hash multiplication and auxiliary boundary products use unsigned/widened
  arithmetic. Buckets agree where upstream's arithmetic is defined. SpCraft
  also avoids upstream's narrowing unsigned loop casts.
- Thread-count discovery uses omp single. Each subsequent team is capped by
  the captured count with num_threads, preventing out-of-bounds scratch access
  if dynamic team sizing changes. Fixed-team comparisons use matching teams.
- Scoped vectors own scratch instead of manual new[]/delete[]. Some arrays
  receive value initialization before the explicit upstream-style assignments.
  Their unused contents never determine a count, lookup, hash probe or value.
  Destruction order and allocator API are not claimed identical.
- CooMatrix follows SpCraft's move-only malloc/free ownership contract and
  explicitly constructs/destroys tuple objects. The tuple element type and
  scalar-component initialization match upstream's tuple array.
- The baseline fixes sort=true and clearA=clearB=false, and uses one value type.
  It does not implement the upstream unsorted-output or operand-deletion modes.

The code retains the upstream license notice. No unordered containers or
try/catch occur in the kernel. CSC/CSR conversion and input canonicalization
are explicit caller responsibilities; no alternate implementation is selected.

## Verification without timing

[test_mtspgemm.cpp](../../tests/test_mtspgemm.cpp) is a plain
correctness executable. With SPCRAFT_USE_COMBBLAS=ON it calls the actual pinned
CombBLAS helpers and kernel and checks:

- Every auxiliary boundary and every materialized A-column range, exercising
  both lookup branches, empty buckets, absent keys and irregular column IDs.
- Full flop counts, symbolic distinct counts, symbolic/numeric capacities,
  every work-prefix entry and every output-prefix entry.
- Every output row/column and the exact double value bits.
- At one thread, the complete Multiply/Add call sequence and both arguments,
  detecting extra first-insertion Add calls and reversed accumulation operands.

Sixteen canonical products are compared at 1, 2 and 4 threads. They include
variable column sizes for scratch growth/reuse, cancellation, stored zeros,
an entirely empty candidate output column, a dense-column input and the
64-product/two-output-row case. The latter now has symbolic capacity 64 on
both sides. The -0.0*1.0 case retains negative zero on both sides.

Without CombBLAS, the same executable checks a hand-computed rectangular
product, empty output shape, dimension mismatch, signed zero, Boolean and
min-plus semantics, index/offset width variants, exact-limit and overflowing
prefixes, and maximum-width lookup rejection before increment overflow.

```sh
cmake --build build_spgemm --target test_mtspgemm
build_spgemm/tests/test_mtspgemm
```

The optional comparison dependency is linked only to the test/benchmark. The
library baseline itself does not depend on CombBLAS, MPI or a source checkout.

Validation completed: all 15 GCC Debug/ASan CTest executables passed. The
upstream-enabled mapped test passed all 48 comparisons against the pinned
checkout. Clang/OpenMP and GCC without OpenMP passed the mapped local checks
under AddressSanitizer and UndefinedBehaviorSanitizer. The updated comparison
executable built; no timing experiment was run for this change.

## Single implementation validation

`test_mtspgemm` is the one kernel test executable. It includes the direct
upstream checks above when CombBLAS is enabled, plus an independent dense
arithmetic/Boolean-pattern reference over 12 canonical products at 1, 2 and 4
threads, noncommutative multiplication, Boolean and min-plus rings, structural
zeros, exact uint8 output limit and full-work overflow. Compile-time checks
accept DCSC inputs and reject CSR/CSC calls to this entry point. The Python tests
exercise explicit CSR canonicalization, duplicate cancellation, empty shapes
and dimension mismatch through the same kernel.
