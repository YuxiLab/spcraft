# CSC/DCSC correctness audit

**Historical audit of removed implementations.** The single maintained kernel is
`OmpHashSpGEMM` in [mtSpGEMM.h](../../include/kernel/mtSpGEMM.h); its current
[source mapping](SourceMapping.md) defines the supported contract. Old names,
line references and checksums below describe the recorded experiments.


The general-kernel line tables below describe the reviewed source snapshot
before COO consolidation. Current native outputs use `CooMatrix::entries`; the
COO ownership review below and [mapped baseline mapping](SourceMapping.md)
describe the current representation. Historical source line numbers are retained
for the recorded general-kernel experiment.

This audit covers the general-purpose column kernels. The separate
CombBLAS-mapped baseline and its direct upstream checks are documented in
[SourceMapping.md](SourceMapping.md).

This is a source review of [DcscMatrix](../../include/core/DcscMatrix.h), its
[implementation](../../include/core/DcscMatrix-inl.h), the
removed column kernels, and the
removed column comparison driver. Adjacent lines that perform one
operation are grouped below; every executable statement is covered. Timings
are not evidence for any of the arguments here.

## Contract and conclusions

The column implementation is native: it does not call the CSR kernel, transpose
operands, or construct full CSC offsets inside the DCSC multiplication.
Both storage types use the same templated column expansion with a different
compile-time left-column lookup. Compressed output matches the input type; spgemm_tuples_openmp selects a native array of std::tuple<IT,IT,NT> at compile time.

Inputs must be well-formed, immutable host matrices. CSC has `n+1` monotone
offsets; DCSC has `nzc+1` strictly increasing offsets for its nonempty columns
and sorted, unique, in-range `col_id`. Both start at zero and end at `nnz`.
Row indices must lie in `[0,m)`. Duplicate and unsorted row entries are allowed.
The default empty object has no buffers and is supported. Raw pointer
constructors borrow valid buffers; they do not validate them.

`IT` and `OT` are supported integral index and offset types. `NT` follows the
host container's raw-storage contract. Semiring operations must be thread-safe
and nonthrowing. There is no `try`/`catch` in either kernel. Allocation and
sizing failures outside OpenMP propagate normally; worker allocation failure
is not recoverable through this API under OpenMP.

No correctness defect was found under that contract. Particular hazards were
handled explicitly in the implementation: terminal-offset sizing, integer
bucket arithmetic at maximum dimensions, absent left columns, output-column
removal, full hash occupancy, and semiring multiplication order. This is a
manual invariant review with regression evidence, not a machine-checked proof.

## DCSC container, line by line

The declaration initializes all pointers and counts and deletes copying. Its
pointer constructor is the borrow path; `Allocate`, `Clone`, `FromCsc`, and
`ToCsc` produce independent owning storage. The four arrays occupy
`(nzc+1)*sizeof(OT) + nzc*sizeof(IT) + nnz*(sizeof(IT)+sizeof(NT))` bytes.
Logical `n` does not appear in the allocation sizes.

Line numbers below refer to `DcscMatrix-inl.h`.

| Lines | Operation and correctness argument |
|---|---|
| 1–18 | Includes and declaration of the private allocator. Only `Allocate` calls it, after rejecting negative counts. |
| 20–27 | Compute every array size before allocating. `nzc` is checked before narrowing; the terminal `+1` is checked before addition, then the entire pointer-array byte count is checked again. Neither a byte-sized offset nor a maximum-width index can bypass sizing. |
| 28–31 | Allocate with `calloc` only. The offset array always has at least one element. Zero-length column, row, and value arrays use null pointers deliberately. |
| 32–36 | Any allocation failure frees every partial buffer before throwing. No member of an existing matrix has changed. |
| 39–49 | `SafeDelete` frees all four buffers only for owners. It takes pointers by value, so callers overwrite or reset their own members. |
| 51–64 | The pointer constructor records shape/counts and sets `memowned=false`; destruction cannot free caller storage. |
| 66–79 | Move construction transfers all four pointers and every metadata field, including ownership, then resets the source. Moving a view remains a view. |
| 81–98 | Move assignment guards self-move, frees existing owned storage, takes all source state, then resets the source. Pointer/count assignments cannot throw. |
| 100–104 | Destruction uses the same ownership-aware deallocator. |
| 106–122 | Allocation validates signed dimensions and offsets conditionally. It enforces `nzc <= n`, `nzc <= nnz`, equivalence of zero nnz and zero stored columns, and a positive row dimension for nonempty storage. Duplicate rows mean `nnz` need not be bounded by `m*n`. |
| 123–130 | Build replacement first, then free the old buffers and publish the new state. Thus an invalid or unrepresentable allocation preserves a populated matrix. Allocation on a view acquires new ownership without freeing the borrowed buffers. |
| 133–144 | Clone allocates with the same shape and counts, then copies all live arrays. Successful allocation already established the representability of `nzc+1`. A default empty object skips null input buffers; its clone still owns one terminal offset. |
| 147–157 | Reset nulls all pointers/counts and releases ownership without freeing. It is correct after stealing storage or for a view; an owner must retain and free its detached buffers if calling Reset directly. |
| 159–167 | FromCsc counts structurally nonempty columns. The count is at most `source.n` and at most `source.nnz`, so it fits IT and satisfies allocation's column-count contract. |
| 168–175 | Traverse logical columns in increasing order and copy only nonempty headers. Empty CSC columns consume no entries, so retained start offsets remain contiguous. The terminal offset is the original nnz. |
| 176–180 | Copy all row/value entries, preserving their ordering, duplicates, and stored zeros. No numerical operation occurs. |
| 183–189 | ToCsc explicitly allocates the full logical width; unlike the DCSC kernel, this conversion can fail for an enormous shape. Initial offset and slot are zero. |
| 190–194 | At each logical column, `offset` is the number of entries preceding it. If the column is stored, advance to its ending offset; otherwise retain the current offset. Strictly increasing column IDs ensure each stored column is used once. |
| 195–199 | Row/value arrays already have CSC's column-major order and can be copied unchanged. Default and shaped empty objects also work. |

## Column kernels, line by line

Line numbers refer to the current `mtSpGEMMColumn.h` (411 lines).

| Lines | Operation and correctness argument |
|---|---|
| 1–29 | Includes, guarded OpenMP declarations, and compile-time storage adapter declaration. |
| 31–52 | CSC borrows immutable input, maps each column to itself and reads its adjacent offsets. Valid B row IDs and A.n==B.m prove lookup indices are in range. |
| 54–65 | DCSC stores a reference. Empty input needs no lookup. Sorted unique in-range IDs with nzc==n imply col_id[k]==k, so dense-column lookup needs no auxiliary allocation. |
| 66–75 | Integer ceiling division computes bucket width and count without n+1 overflow. nzc>0 and n>=nzc establish nonzero divisors. Auxiliary element counts and the terminal +1 are checked before allocation. |
| 76–92 | Monotone scan writes the first column slot in each bucket, including terminal nzc. Size/Index report the stored-column mapping. No logical-width allocation occurs. |
| 93–106 | Empty lookup returns an empty range. Dense lookup directly accesses valid offsets. Otherwise lower_bound searches only the bucket containing the query; absent keys and empty buckets return empty ranges without dereferencing end. |
| 108–114 | CSC output allocates exactly nnz and copies the n+1 candidate prefixes. |
| 116–133 | DCSC counts and emits only positive-length output columns in B storage order. Zero-length columns occupy no entries, so retained offsets remain contiguous. Terminal offset equals nnz. |
| 135–154 | Checked prefix helper uses the serial path for one thread or fewer than 4096 columns. Before each sum, count<=max(OT)-nnz proves representability before narrowing. Numeric pair capacity is validated separately from symbolic key capacity. |
| 155–162 | Nonempty parallel input is divided into at most min(threads,columns) nonempty contiguous blocks. Ceil division avoids addition overflow; block arrays are proportional to stored columns. OpenMP names every shared constant and buffer. |
| 163–179 | Each block writes only offsets[begin+1..end] and its own totals/maxima/overflow entries. It checks every local sum before narrowing; an overflow saturates at max(OT) and records a byte flag. No exception is thrown by this worker loop. The barrier publishes all blocks. |
| 180–188 | Calling thread rejects any block overflow and checks every addition of block totals. It replaces each total with that block's exclusive base and finds the maximum column count. All global offsets are thus <=max(OT). |
| 189–193 | Capacity is monotone in count. Validating its maximum once proves every following bit_ceil result, byte count and vector max_size is valid. Zero counts will not call bit_ceil. |
| 194–210 | Independent blocks add their checked bases to local offsets and compute numeric capacities. A local prefix plus its base is <=the checked global total, so neither unsigned arithmetic nor narrowing can overflow. Each offset and capacity is written once. |
| 212–217 | Tuple output allocates exactly nnz live tuple objects with the product shape. It does not allocate compressed output buffers or perform conversion. |
| 219–234 | Input/output storage are compile-time parameters. Assert semiring and integral types; reject mismatched/negative dimensions before constructing lookup. |
| 235–250 | B iteration uses B.nzc for DCSC and B.n for CSC. Counts/capacities are checked for byte-size representability; offsets check the extra terminal element before adding it. Default empty buffers are never dereferenced. |
| 251–259 | Positive OpenMP max-thread count and a lower-bounded chunk produce a legal dynamic schedule; no-OpenMP uses one thread. Bound is A.m, the maximum distinct output rows. |
| 261–273 | For every candidate B column sum referenced A-column lengths, saturating at A.m via min(length,A.m-bound). This cannot overflow, even at maximum IT. It bounds distinct reachable rows, including duplicate paths. |
| 274–276 | Validate symbolic key capacities before worker allocation; empty expansions get zero capacity. |
| 278–288 | Each symbolic iteration owns its hash vector and skips zero capacity before forming a mask. max(IT) is an impossible valid row because row<A.m<=max(IT). |
| 289–302 | Expand A(:,k) for each stored B(k,j), including duplicates. Stride-one probing increments count only on first insertion. No values are read in this phase. The exact count is published at the loop barrier. |
| 304–307 | Checked prefix produces disjoint slices and checked numeric capacities, then output policy allocates the final representation. |
| 309–318 | Each numeric iteration owns real pair<IT,NT> scratch, including NT=bool without proxy references. All accumulators begin at the semiring identity. |
| 319–331 | Repeat the immutable symbolic pattern. Every input pair contributes once as Add(acc,Multiply(A.val[a],B.val[b])). Multiplication order is preserved for noncommutative rings. |
| 332–337 | Compaction destination never exceeds source slot, so unread buckets cannot be overwritten. Sorting moves complete pairs and visits only the occupied prefix. |
| 338–353 | Tuple output writes (row, logical B column, value), using B.col_id for DCSC. Compressed output writes row/value arrays. Each column writes exactly its symbolic count; dest reaches, never exceeds, its end offset, including max(OT). |
| 354–358 | Return transfers owning output; temporary lookup/count/capacity/prefix buffers are released by their owners. |
| 360–390 | Existing CSC/DCSC entry points enforce the semiring type, document contracts and dispatch to the same implementation. |
| 392–409 | Tuple entry point restricts input to CSC/DCSC, checks the semiring type and selects native COO output at compile time. No conversion is hidden in the call. |

### COO entry ownership

[CooMatrix.h](../../include/core/CooMatrix.h) now owns native coordinate outputs.
The separate tuple container has been removed.

| Operation | Correctness argument |
|---|---|
| Entry storage | `Entry = std::tuple<IT,IT,NT>` and raw `Entry* entries`; tuple component type and initialization match the native reference. General COO makes no sorting assumption; the producing kernel guarantees column/row order. |
| SafeAllocate | Check alignment and `count*sizeof(Entry)` before allocation. Guard malloc storage while `uninitialized_value_construct_n` starts live entries. A throwing constructor destroys partial entries through the standard algorithm and the guard frees the allocation. |
| Release | Only an owner destroys live entries and frees storage; views leave the borrowed array untouched. |
| Pointer constructor and moves | Borrow a live Entry array. Moves transfer pointer, shape and ownership, then reset the source. Assignment guards self-move and releases only its previously owned allocation. |
| Allocate | Reject negative shapes/counts and nonzero nnz with a zero extent. Build replacement storage before committing metadata or releasing old storage. |
| Clone | Allocate and copy complete entries into an independent owner. Default and shaped-empty clones work. |
| Reset | Clear pointer, shape and ownership without freeing, for moved-from objects and views. |
| ToCsr | Validate coordinates; count and prefix rows; scatter complete entry components. Preserve duplicate entries, stored zeros, and within-row order without mutating input. |
| FromMatrixMarket | Check header dimensions before narrowing and final nnz before narrowing to OT. Parse through temporary vectors and copy into the Entry array; temporary vectors do not back the matrix. |
| Python boundary | Copy incoming row/column/value arrays into entries and project each outgoing component into a fresh contiguous NumPy-owned array. No NumPy array aliases C++ storage. |

`test_coo_matrix` covers the combined lifetime and conversion contract, including
construction failure with nontrivial values, overflow, borrowing, moves, cloning,
empty shapes, duplicates, unsigned Boolean entries and Matrix Market input.
The native kernel tests continue to check coordinate order and numerical values.

**Full-table termination:** If a new key is encountered, fewer distinct keys
have been inserted than the bound used to allocate the table, hence an empty
slot exists. Once a table is full, every remaining key must already be present.
Stride-one linear probing visits all slots, so that existing key is found.
The exact symbolic count establishes the same argument for numeric probing.

**Parallel ownership:** Inputs, lookup buckets, capacities, and numeric offsets
are read-only in worker loops. Each worker owns a separate scratch vector and
writes only its own count or output slice. Implicit barriers separate expansion,
symbolic, prefix/output allocation, and numeric work. Every OpenMP region uses
`default(none)` with explicit shared variables; the serial path uses the same
loop bodies. This does not validate arbitrary concurrent mutation by callers.

## Shared hash helpers and existing CSR path

`SpGEMMHash.h:16–35` is the previously reviewed capacity helper: zero is a skip,
nonzero capacity starts at 16, checked doubling preserves powers of two, and
byte sizing plus vector max_size are checked before parallel allocation.
Lines 37–44 hash after conversion to unsigned size_t, then mask; original keys
remain full-width for equality. Unsigned wrap changes collisions, not identity.
These helpers were moved unchanged out of `mtSpGEMM.h`; the existing CSR
algorithm and API remain unchanged. Its [earlier audit](CorrectnessAudit.md)
retains the exact reviewed source snapshot.

## Benchmark correctness and comparison scope

`SpGEMMColumnBenchmark.cpp` canonicalizes input once using Eigen, builds CSC and
DCSC copies and CombBLAS DCSC copies before timing, and computes A*B in the same
orientation for every backend. It checks the complete CSC offsets, both DCSC
column-ID/offset arrays, all row indices, and every value against independent
Eigen output at each thread count. The empty output avoids dereferencing
CombBLAS's absent DCSC data. Nonfinite actual values are rejected by Verify;
inputs in the measured suite and Eigen references are finite.

A rectangular 2x3 by 3x2 benchmark passed in FP32/FP64 with int32/int64 at one
and four threads. The hypersparse smoke input passed full verification with
64-fold embedded logical dimensions. Embedding maps both row and column IDs
by the same stride and leaves values unchanged, so products at mapped IDs are
identical and every other logical row/column is empty.

The primary comparison uses identical DCSC inputs and native sorted tuple outputs.
Both native clocks include symbolic, numeric, sorting, initialized tuple allocation,
and destruction; neither converts the output. CooMatrix uses malloc/free with
explicit tuple lifetimes, while CombBLAS uses new[]/delete[]. This allocator API
difference is disclosed; both store exactly std::tuple<IT,IT,NT> in the same binary.
Both also have a DCSC-output measurement, with CombBLAS's parallel five-argument
compression constructor included. CSC is an additional format diagnostic.

The unmodified upstream limitations in the prior audit still apply. The new
driver additionally bounds signed hash keys, excludes maximum logical inner
dimension (upstream uses n+1), and conservatively bounds total work before
signed symbolic capacity doubling. It cannot repair upstream's unsynchronized
numThreads writes; independent reference checks remain essential.

The five-backend Williams design starts with 0,1,4,2,3, adds each residue modulo
five, then repeats those five rows reversed. Over ten rounds each treatment
occupies each position twice and each directed adjacency occurs twice. A time
cutoff can truncate the balance; sample counts and every raw round are retained.
All output patterns, including native tuple column IDs, are independently checked
against Eigen before timing at every thread count.

## Validation

- All 14 Debug/ASan CTest executables passed, including existing CSR regressions.
- Both new test executables passed with Clang/OpenMP and GCC without OpenMP
  under AddressSanitizer and UndefinedBehaviorSanitizer.
- Container checks cover default and shaped empties, owned/borrowed storage,
  moving owners and views, assignment, self-move, deep cloning, Reset, failed
  replacement preservation, invalid counts, CSC roundtrips, and huge shapes.
- Column tests cover hand-computed rectangular output, duplicates/cancellation,
  structural zeros, empty columns and inner dimensions, OR-AND, min-plus,
  noncommutative composition, Eigen FP32/FP64 comparisons, collisions with
  wraparound and full tables, uint8 output overflow and exact 255 entries,
  missing left columns, and signed/unsigned maximum logical dimensions.
- Additional lookup checks cover empty buckets, multiple keys in one bucket,
  absent keys within populated buckets, default null-buffer products, and
  negative dimensions before lookup construction.
- The BigRed script runs both new test executables before measurement.

The tuple extension adds ownership/empty/overflow tests and runs the existing
hand cases, Boolean/min-plus/noncommutative cases, collision tests and Eigen
comparisons through both CSC-to-tuples and DCSC-to-tuples. A 4096-column case
checks parallel prefixes ending at exactly 65535 entries, global overflow across
blocks, local-block overflow, and tuple dimension mismatch.

The native container also tests a throwing nontrivial value constructor after
partial replacement construction: previously constructed replacement entries
are destroyed, their malloc buffer is freed, and the existing owning matrix
remains intact. Maximum signed/unsigned DCSC logical dimensions are exercised
through tuple output without constructing a logical-width offset array.

The optional native-only measurement selects backend IDs 4 and 2, alternates
AB/BA each round and emits only those two records. It changes timed selection,
not inputs, kernel specialization, verification, ownership or timer boundaries.
The five-path job retains its exact reviewed benchmark source as
`columns-bigred-8168162/reviewed-column-benchmark.txt` beside its source hashes.
