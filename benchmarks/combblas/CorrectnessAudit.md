# Hash SpGEMM correctness audit

**Historical audit of removed implementations.** The single maintained kernel is
`OmpHashSpGEMM` in [mtSpGEMM.h](../../include/kernel/mtSpGEMM.h); its current
[source mapping](SourceMapping.md) defines the supported contract. Old names,
line references and checksums below describe the recorded experiments.


This review covers [the production kernel](../../include/kernel/mtSpGEMM.h),
the pinned CombBLAS reference, and the comparison driver. It examines the code
and its invariants independently of performance measurements. Timings are not
used as evidence of correctness.

## Findings and fixes

1. **Row scratch counts were narrowed too early.** Casting `A.m` to `size_t`
   before checking it can truncate a wider index on a narrower address space.
   Both scratch arrays now use `CheckedElementCount` before allocation. The
   regression requests an unrepresentable row count and requires failure
   before any input buffer is accessed.
2. **A byte-count check alone did not establish vector representability.**
   A power-of-two capacity can fit `size_t` yet exceed `vector::max_size()`.
   The helper now checks both limits outside the parallel loops. A regression
   exercises the limit imposed by the iterator difference type.
3. **The input and semiring preconditions needed to be explicit.** The doc
   comment now states them. This kernel validates shapes and allocation sizes;
   it is not a validator for arbitrary raw CSR buffers.

No arithmetic or indexing defect was found in the row expansion, probing,
accumulation, compaction, or output prefix logic under the contract below.
That conclusion rests on the invariants in this review plus independent
regression checks; it is not a formal machine-checked proof.

## Required contract

- Each operand is valid host CSR: nonnegative dimensions/counts, a row-pointer
  array of length `m+1`, nondecreasing offsets starting at zero and ending at
  `nnz`, and readable column/value arrays for the stored entries. Every column
  lies in `[0, n)`. A default `0x0` matrix needs no input buffers.
- Inputs remain unchanged throughout all phases. Output storage is newly
  allocated. Duplicate and unsorted input entries are permitted.
- `IT` and `OT` are supported integral index/offset types. `NT` is a value type
  supported by the raw-storage CSR container. The provided semiring has the
  stated additive identity and valid operations on the supplied values.
- Semiring operations are thread-safe and do not throw in worker loops.
  There is no `try`/`catch` in the kernel. Worker allocation failures are not
  recoverable exceptions at this API boundary under OpenMP; explicit sizing
  and output-overflow failures are checked outside those loops.

## Line-by-line review of the SpCraft kernel

Line numbers refer to the [reviewed 228-line CSR snapshot](../results/combblas/spgemm/bigred-8165816/reviewed-csr-kernel.txt),
SHA256 `c1b37af7d6fccf8be113be04cab093c9625382da5c5dcf37a6ee2d6d82c61326`.
The hash helpers have since moved unchanged to `include/kernel/SpGEMMHash.h`;
the CSR algorithm is unchanged. The new CSC/DCSC code has a separate
[line-by-line audit](ColumnCorrectnessAudit.md).
Consecutive lines implementing one operation are grouped; every executable
statement is covered.

| Lines | Operation | Correctness argument |
|---|---|---|
| 1–22 | Includes, namespaces | OpenMP declarations are conditional, so the header also compiles without OpenMP. Allocation guarding is included directly. |
| 24–29 | Empty hash capacity | A zero key bound returns zero. Both parallel passes skip these rows before forming a mask or indexing a table. |
| 30–37 | Capacity rounding | The initial capacity is 16, a power of two. Each multiplication by two is preceded by a `size_t` overflow check. On success, capacity is a power of two at least as large as the requested bound. |
| 38–42 | Physical table sizing | Byte sizing is checked before narrowing/allocation. The additional vector limit establishes that the actual backing vector and its iterator range can represent the capacity. Calls occur outside OpenMP loops. |
| 45–51 | Hash arithmetic | Conversion to unsigned `size_t` occurs before multiplication by 107, so multiplication cannot cause signed-overflow UB. Masking retains the bucket bits. Original, full-width keys are stored and compared, so collisions or discarded upper hash bits do not change identity. |
| 56–72 | API contract | Sorted output and retained structural identities are deliberate. Input immutability is essential because numeric capacity depends on the symbolic result. |
| 73–80 | Type checks | `SemiRing::ValueType` must match `NT`; index and offset types must be integral. The compiler does not prove semiring axioms. |
| 82–89 | Shape checks | Incompatible dimensions and negative signed dimensions fail before any buffer access. Unsigned instantiations do not compile a meaningless negativity branch. |
| 91–95 | Scratch arrays | Row counts are checked before conversion to an addressable element count. Each row owns one count and capacity slot. `B.n` is representable in `uintmax_t` for the supported standard index types. |
| 96–105 | Scheduling grain | The chunk size is at least one and at most 256. Division uses a positive thread count and avoids an overflowing thread-count product. Scheduling affects row assignment, not the order of operations inside a row. |
| 107–118 | Expansion bound | Before each update, `0 <= bound <= B.n`. The update adds at most `B.n-bound`, preserving that invariant without overflowing. The number of distinct reachable columns is at most both the scalar-product count and `B.n`. |
| 119–124 | Saturation and storage | Once the bound reaches `B.n`, additional paths cannot increase the required number of distinct keys. Breaking early is safe. A zero bound remains zero for an empty product row. |
| 125–128 | Symbolic capacity | Rounding the bound produces enough slots for every distinct reachable column. Unrepresentable capacities fail before worker allocation begins. |
| 130–139 | Symbolic row ownership | The parallel loop writes distinct count elements. Empty rows skip table construction. Implicit loop barriers separate all phases. |
| 140–145 | Sentinel and mask | `max(IT)` cannot be a valid column because valid columns are strictly less than `B.n <= max(IT)`. This works for signed and unsigned indices, even when `B.n == max(IT)`. Nonempty capacities make `capacity-1` a valid mask. |
| 146–150 | Structural expansion | For each stored `A(i,k)`, every stored entry of `B(k,:)` contributes its column key. Values are not read in this phase. Valid CSR guarantees all row and entry accesses are in range. |
| 151–157 | Probing and counting | A probe advances by one modulo a power-of-two capacity and therefore visits every slot. An existing key is found even in a full table. Before a new key is inserted, capacity exceeds the number already inserted, so an empty slot exists. Only first insertion increments the count. |
| 158–161 | Exact symbolic result | Each distinct output column was encountered and inserted once; the stored count is exact, regardless of duplicates or input ordering. |
| 163–171 | Output nnz check | The maintained invariant is `nonzeros <= max(OT)`. Checking `count <= max(OT)-nonzeros` before adding prevents overflow of both the accumulator and its later narrowing. Equality with the limit is valid. |
| 173–175 | CSR allocation | Allocation uses the checked total and exact output shape. Container ownership handles cleanup if subsequent serial sizing fails. An empty output still has an allocated row-pointer array. |
| 176–181 | Row prefix and numeric capacity | Each row count and every nonnegative partial sum are bounded by the checked total. Therefore narrowing and prefix addition fit `OT`. Numeric capacity is derived from the exact count and independently checked for the larger key/value entry type. |
| 183–195 | Numeric row ownership | Every worker has a private vector of real key/value pairs, including when `NT=bool`. Accumulators start at the semiring identity. Each worker writes only its row's disjoint CSR slice. |
| 196–203 | Numeric expansion/probing | The same immutable structure is traversed as in the symbolic phase. Consequently no more distinct keys can arrive than the exact capacity bound allows. The symbolic termination argument applies again. |
| 204–206 | Numerical contribution | For each path, the update is `Add(accumulator, Multiply(A_value, B_value))`. Operand order is preserved even for noncommutative multiplication. Every stored input-entry pair contributes exactly once, while repeated output keys accumulate in one slot. |
| 209–214 | In-place compaction | Before processing slot `s`, the destination index equals the number of occupied slots already seen, hence is at most `s`. A copy cannot overwrite an unread future slot. No hash lookup occurs after compaction. The final count equals the symbolic count. |
| 215–216 | Sorting | Only the occupied prefix is sorted, using column keys alone. Whole pairs move together, preserving the key/value association. Table/vector sizing makes the iterator range representable. |
| 217–223 | Output write | Exactly the symbolic number of entries is written from `row_ptr[row]` to `row_ptr[row+1]-1`. The final increment reaches the end offset without exceeding it, including when the end equals `max(OT)`. Empty rows write nothing. |
| 225–228 | Return/lifetime | Worker vectors are destroyed in their iterations; row scratch is local RAII storage. The returned CSR transfers ownership using the container's move semantics or return-value elision. |

The key termination detail is that **a completely full table is allowed**.
Once full, every remaining key must already exist, because capacity bounds the
total number of distinct keys. Searching a full table for a key absent from
the symbolic pattern would violate the immutable, valid-input contract.

## Review of the pinned CombBLAS reference

Reference: [`mtSpGEMM.h` at `2381a1f`](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/mtSpGEMM.h).

| Lines / routine | Review result |
|---|---|
| 18–69, `prefixsum` | Two identical static partitions and a barrier connect thread-local prefixes to thread offsets. Integer totals must fit the selected type. |
| 464–523, `LocalSpGEMMHash` setup | Handles zero operands, obtains expansion counts, runs symbolic counting, and allocates tuple output. |
| 531–563 | Maps input columns and rounds exact output counts to power-of-two numeric capacities. |
| 568–603 | Sentinel initialization, linear probing, duplicate reduction. Uses `add(new, accumulated)`, valid for semiring addition. |
| 607–635 | Occupied pairs compact safely before sorting; sorted tuples preserve column grouping. |
| 639–651 | Optional operand deletion is disabled by the driver; temporary arrays are released. |
| 807–934, `estimateNNZ_Hash` | Counts keys with linear probing; thread-indexed symbolic scratch is reused. |
| 1058–1141, `estimateFLOP` | Computes structural work counts used for symbolic sizing. |

Code-level limitations: signed `key*107`, signed capacity doubling, and unchecked
`IT` prefix totals can overflow. Several parallel regions write `numThreads`
without synchronization; identical writes are still a data race. The checkout
is unmodified. Benchmark work counts are checked, and Eigen provides the
independent reference; agreement with CombBLAS alone would be insufficient.

## Comparison-driver correctness

- Both operands are canonicalized outside timing, then independently multiplied
  by Eigen. Input conversions and preparation are excluded from both clocks.
- CombBLAS receives **`B^T, A^T`**, in that order. Its column-sorted result is
  `C^T`, which has the same linear entry ordering as SpCraft's row-sorted `C`.
  This benchmark uses plus-times scalar arithmetic, where reversing scalar
  multiplication operands is valid. This comparison transformation is not
  claimed for arbitrary noncommutative semirings.
- At every thread count, the driver checks total nnz, all SpCraft row pointers,
  both backends' column/tuple indices, and every value against Eigen. It rejects
  nonfinite output and numerical errors beyond the dtype tolerance. Both
  timing paths allocate and destroy their native output.
- The reference's work prefix uses `IT`; the driver checks the complete work
  count before calling it. The benchmark uses matching index/offset widths.
- A hand-checked rectangular `2x3` by `3x2` product was also run through the
  driver in both precisions and both index widths. This checks operand order
  and transpose handling that an `A*A` experiment alone would not establish.

## Regression evidence

The dedicated `test_mtspgemm` covers a hand-computed rectangular product,
unsorted duplicates, cancellation and stored zeros, empty rows and dimensions,
OR-AND, min-plus, noncommutative relation composition, invalid dimensions,
overflow and exact-limit offsets, hash capacity limits, full collision chains
with wraparound, largest valid signed/unsigned column indices, and float/double
products checked against Eigen. The Eigen case has 97 rows, spanning multiple
scheduling chunks.

GCC/OpenMP, Clang/OpenMP, and serial builds are used for regression validation,
with AddressSanitizer and additional undefined-behavior checks. The cluster job
also runs the dedicated test before collecting benchmark samples.
