# General-purpose SpCraft kernel ↔ CombBLAS source mapping

**Historical audit of removed implementations.** The single maintained kernel is
`OmpHashSpGEMM` in [mtSpGEMM.h](../../include/kernel/mtSpGEMM.h); its current
[source mapping](SourceMapping.md) defines the supported contract. Old names,
line references and checksums below describe the recorded experiments.


This is a historical source snapshot from before tuple-output ownership was
consolidated into `CooMatrix::entries`. The recorded old names, line numbers and
checksums describe that snapshot. Current storage is documented in the
[COO contract](../../README.md#sparse-matrix-storage); the mapped baseline has its
own [current mapping](SourceMapping.md).

The general-purpose spgemm_tuples_openmp DCSC-to-tuples kernel is **not an exact implementation of
CombBLAS's logic**. It shares column expansion, symbolic/numeric hashing and
sorted tuple output, but differs in lookup, work estimation, table sizes,
scratch lifetime, prefix construction, scheduling and numeric insertion.
Equal products do not establish equal intermediate operations.

This review compares `spgemm_tuples_openmp<Ring>(A,B)` for DCSC inputs with
`LocalSpGEMMHash<SR,NT>(A,B,false,false,true)`. It examines source and small
structural witnesses; no timing result is used as evidence.

Reference revision: `2381a1f7a028b9f56d8531c0b7b157ae2ab27572` in
[hongyx11/CombBLAS](https://github.com/hongyx11/CombBLAS/tree/2381a1f7a028b9f56d8531c0b7b157ae2ab27572).
The reference is **LocalSpGEMMHash**, not the heap-based LocalSpGEMM or the
separate hybrid kernel elsewhere in the same header.

## Storage and variable correspondence

| CombBLAS | SpCraft | Meaning |
|---|---|---|
| `A.getnrow(), A.getncol()` | `A.m, A.n` | Logical shape |
| `Adcsc->nzc` | `A.nzc` | Number of stored nonempty columns |
| `Adcsc->cp` | `A.col_ptr` | Offsets indexed by stored-column slot |
| `Adcsc->jc` | `A.col_id` | Logical ID of each stored column |
| `Adcsc->ir` | `A.row_id` | Row IDs in column storage order |
| `Adcsc->numx` | `A.val` | Values in the same order |
| `i` in the outer loop | `col` | Stored B-column slot, not necessarily logical column ID |
| `Bdcsc->jc[i]` | `B.col_id[col]` | Logical output column |
| `j`, then `Bdcsc->cp[i]+j` | `b` | Entry of B in that stored column |
| `colinds[j].first/second` | `left.Range(B.row_id[b])` | Start/end offsets of A(:,k) |
| `k` in the inner expansion | `a` | Entry offset in the selected A column |
| `key` | `row` | Output row ID |
| `colnnzC[i]` | `counts[col]` after symbolic | Exact number of distinct output rows |
| `colptrC[i]` | `offsets[col]` | Beginning of this output-column slice |
| `globalHashVec` | `keys` or `table` | Symbolic keys or numeric row/value pairs |
| `tuplesC[curptr]` | `result.tuples[dest]` | Exactly `std::tuple<IT,IT,NT>`: (row,column,value) |

Both inputs are DCSC, not a CSR reinterpretation or a transposed product.
The shared domain requires well-formed immutable structure, sorted unique DCSC
column IDs and canonical B row lists. Upstream counts, hash products, capacities,
int-sized prefix lengths and unsigned casts must be representable, and its
float-based lookup must resolve the intended buckets. SpCraft's broader large-
index and duplicate/unsorted-input support is not automatically shared by
upstream. The concrete witnesses below use tiny canonical matrices within
these limits.
SpCraft separates index `IT` and offset `OT`; compare with `OT=IT` to match
CombBLAS. CombBLAS permits distinct NT1, NT2 and NTO, whereas this SpCraft API
uses one NT and asserts `SemiRing::ValueType==NT`.

## Operation-by-operation mapping

SpCraft line numbers refer to the current 411-line
`mtSpGEMMColumn.h` (removed).
Upstream `mt` denotes
[mtSpGEMM.h](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/mtSpGEMM.h),
and `dcsc` denotes
[dcsc.cpp](https://github.com/hongyx11/CombBLAS/blob/2381a1f7a028b9f56d8531c0b7b157ae2ab27572/include/CombBLAS/dcsc.cpp).

“Same” below means the stated operation matches on the shared valid-input
range. “Equivalent” means its mathematical result matches but its mechanism
or executed operations differ. “Different” identifies an intermediate or
observable behavior that does not match.

| Step | CombBLAS source and logic | SpCraft source and logic | Assessment |
|---|---|---|---|
| Entry contract | mt 464–481: obtains shapes, returns empty tuples immediately if either operand is zero | 225–250: validates dimensions, builds metadata; no equivalent global nnz shortcut | Different control flow; equivalent empty output shape/pattern |
| DCSC column lookup | mt 483–487; dcsc 1043–1071: constructs auxiliary buckets using float-based ceil((n+1)/nzc) | 54–106: integer ceil(n/nzc), different bucket boundaries; bypasses buckets when nzc==n | Different |
| Lookup inside a B column | dcsc 1363–1425: FillColInds materializes a range per B entry; uses sorted set intersection when nzc/nind<4, otherwise AuxIndex | 93–104, 267–270, 289–291, 319–321: independent Range calls; dense direct offsets or bucket lower_bound | Equivalent ranges on canonical shared inputs; different algorithm |
| Bucket search | dcsc 1144–1155: std::find, with a float round trip for the bucket index | 97–103: integer bucket index and std::lower_bound | Different search procedure |
| Lookup construction count | mt 487,500,505 and helper default arguments: outer lookup plus fresh lookups in estimateFLOP and estimateNNZ_Hash | 234: one lookup reused by all phases, or none for fully populated columns | Different |
| Structural work | mt 1058–1135: flopC[i] is the complete sum of referenced A-column lengths | 261–273: counts[col] is that sum saturated at A.m; traversal can stop early | Different intermediate count |
| Work prefix | mt 501–503: builds flopptr and reads total work; frees it at the end | No counterpart | Different; this prefix does not determine the sorted product |
| Symbolic table capacity | mt 883–893: power of two >=max(16,flopC[i]), including 16 for zero work | 274–286 and SpGEMMHash.h: power of two >=max(16,min(work,A.m)); zero work skips allocation | Different table size and possibly different probe sequence |
| Symbolic scratch ownership | mt 852–853,892–900: per-thread range/key vectors reused across columns, growing when necessary | 282–286: a fresh key vector per candidate column | Different lifetime/reuse |
| Symbolic initialization | mt 902–905: first ht_size keys set to IT(-1) | 285–286: keys initialized to max(IT) | Equivalent sentinel purpose; different signed representation |
| Hash function | mt 913–914: signed key*107 masked by capacity-1 | SpGEMMHash.h 38–44: unsigned size_t(key)*107, same mask formula | Same slot only when capacity matches and upstream arithmetic/conversions are representable |
| Symbolic insertion | mt 915–930: existing key does nothing, empty key increments colnnzC, collision advances one bucket | 292–301: skip collisions; insert/count only when slot is empty | Same distinct-key counting and stride-one probing; branch order differs |
| Output prefix | mt 506 and prefixsum at 24–69: per-thread static prefixes, barrier, each thread sums preceding totals | 135–210,305: checked serial fallback or explicit contiguous blocks, calling-thread block scan, second parallel pass; computes capacities too | Equivalent exact prefixes in the common representable range; different partition and algorithm |
| Output allocation | mt 516: new std::tuple<IT,IT,NTO>[nnzc] | 212–217,306–307 and TupleMatrix.h 32–46: malloc plus explicit value construction | Same element type and component initialization for measured scalar types; different allocator/owner API |
| Numeric range scratch | mt 519–547: per-thread colinds vectors pre-sized from nnzA/thread count and reused | 319–320: direct Range lookup per B entry; no colinds vector | Different |
| Numeric table capacity | mt 553–565: round max(16,exact column nnz) to a power of two | 305,314–318: same rule for positive nnz; skip zero-nnz columns | Same positive-column capacity; different zero-column behavior |
| Numeric table initialization | mt 565–572: default-constructed pairs, then keys=-1; numeric values default-initialized, not semiring-initialized | 316–317: keys=max(IT), values=SemiRing::kAdditiveIdentity | Different |
| Product traversal | mt 576–584: B entries in storage order, then the corresponding A range in storage order | 319–329: same nested B-entry/A-entry order | Same for equal resolved ranges |
| Scalar multiplication | mt 584: multiply(A value,B value) | 329: Multiply(A value,B value) | Same operand order, including noncommutative multiplication |
| First insertion | mt 594–598: assign product directly to the new key | 327–329: assign key, then Add(identity,product) | Different executed arithmetic; can differ in signed-zero bits |
| Repeated-key reduction | mt 589–591: add(product,accumulator) | 329: Add(accumulator,product) | Reversed arguments. Equivalent under commutative semiring addition; not an identical call sequence |
| Compaction | mt 609–616: scan buckets ascending and copy occupied pairs into the front | 332–335: same | Same algorithm; sentinel representation differs |
| Sorting | mt 618: sort occupied prefix by row ID | 336–337: same pair-key comparison | Same ordering rule |
| Tuple emission | mt 620–623: tuple(row,B.jc[i],value) at colptrC[i] | 338–347: tuple(row,B.col_id[col],value) at offsets[col] | Same tuple semantics, layout type and sorted order |
| Scheduling | mt 860–863,1107–1110,526–529: parallel for without schedule clause; default is implementation-defined | 262,279,310: explicit dynamic chunks | Different. The upstream build commonly uses static, but source alone does not mandate it |
| Thread count | mt 490–497 and helpers: all workers write a shared numThreads | 253–255: omp_get_max_threads; no shared worker write | Different; SpCraft avoids the upstream data race |
| Cleanup | mt 639–651: optional operand deletion, delete[] scratch, heap-allocated SpTuples wrapper | 355 and TupleMatrix lifetime: scoped scratch and move-only returned owner; inputs remain borrowed | Equivalent input ownership for clearA=clearB=false; different resource lifecycle |
| Sorting/clearing options | mt signature: sort and clear flags | Tuple API always sorts and never deletes inputs | Match only the explicitly selected false,false,true reference call |

## The two algorithms written with the same names

For candidate column j, define

```
F[j] = sum over stored B(k,j) of nnz(A(:,k))
U[j] = number of distinct reachable row IDs
P[j] = sum of U[q] for candidate slots q < j
```

`F` counts scalar products, not distinct output entries. Duplicate paths
increase F without increasing U. Each implementation discovers U with hashing.

CombBLAS:

```
resolve A ranges for the B entries
F[j] = full expansion count
symbolic_capacity[j] = pow2ceil(max(16, F[j]))
hash reachable rows to obtain U[j]
P = prefix(U)
numeric_capacity[j] = pow2ceil(max(16, U[j]))
for product in B-entry / A-entry order:
    find row's slot by linear probing
    if slot already contains row: value = Add(product, value)
    else:                        value = product
compact, sort by row, emit (row, logical_column, value) at P[j]
```

Current SpCraft:

```
bound[j] = min(F[j], A.m), possibly stopping counting early
symbolic_capacity[j] = 0 if bound[j]==0 else pow2ceil(max(16,bound[j]))
hash reachable rows to obtain U[j]
P = checked_prefix(U)
numeric_capacity[j] = 0 if U[j]==0 else pow2ceil(max(16,U[j]))
initialize numeric values to identity
for product in the same B-entry / A-entry order:
    find row's slot by linear probing
    set its row key
    value = Add(value, product)   // also on first insertion
compact, sort by row, emit (row, logical_column, value) at P[j]
```

The tuple output specialization shares the compressed-output implementation;
returning tuples changes allocation and emission, **not** these differences.

## Concrete witnesses

### Different symbolic table with canonical inputs

Let A be 2x32 with both rows stored in every column, and let B be 32x1
with all 32 rows stored once. Every input value is one.

- Full expansion F[0]=64; distinct count U[0]=2; output values are both 32.
- CombBLAS uses symbolic capacity 64.
- SpCraft bounds at A.m=2 and uses symbolic capacity 16.
- For row key 1, the initial symbolic slot is `107 & 63 = 43` in CombBLAS
  and `107 & 15 = 11` in SpCraft.
- Both numeric tables have capacity 16, because both found U[0]=2.

This proves that identical sorted products do not imply identical symbolic
hash operations, even with canonical finite inputs and no overflow.

### Different arithmetic on first insertion

Let A=[-0.0], B=[1.0], using ordinary IEEE double plus-times and the default
round-to-nearest environment, without fast-math.

- Product is -0.0.
- CombBLAS assigns that first product: output retains negative zero.
- SpCraft executes Add(+0.0,-0.0): output is positive zero.

Both witnesses were checked with the actual pinned CombBLAS helpers/kernel
and the actual SpCraft tuple kernel, using GCC 14.3, C++20, -O0, one OpenMP
thread, and no fast-math. The capacity for SpCraft was derived with its actual
capacity helper from min(F,A.m); F and U were obtained from upstream's actual
estimateFLOP and estimateNNZ_Hash. Actual tuple outputs confirmed the sign bits.

```
canonical 2x32 * 32x1: upstream work=64, exact symbolic nnz=2,
SpCraft output nnz=2, upstream symbolic capacity=64,
derived SpCraft symbolic capacity=16
single -0 * 1: upstream signbit=1, SpCraft signbit=0
```

The values compare equal numerically, but their sign bits differ. An Eigen
check using an absolute tolerance cannot establish identical arithmetic here.

### Different input domain

FillColInds can choose std::set_intersection. That branch requires sorted
query row IDs and consumes an intersection match once; it does not preserve
repeated B row queries in the way independent lookups do. SpCraft explicitly
supports unsorted and duplicate row entries. Source equivalence claims must
therefore use canonical sorted, duplicate-free B columns, or explicitly
account for a changed lookup contract.

### Different empty-column work

If a stored B column refers only to absent columns of A, F[j]=U[j]=0.
CombBLAS still uses a minimum 16-entry symbolic and numeric table for that
candidate column. SpCraft skips both. Both emit no tuple for it.

## What exact source alignment would require

The current implementation cannot be labeled an exact CombBLAS baseline.
A faithful baseline needs all of the following, before any optimization:

1. Full F counts and the corresponding symbolic capacities, including the
   zero-column minimum table; retain the work prefix if matching all phases.
2. The same FillColInds range materialization, auxiliary construction and
   lookup-selection rule on the agreed canonical input domain.
3. The same symbolic scratch reuse and initialization, plus numeric range
   scratch lifecycle. Dense-column bypass is a separate optimization.
4. The same prefix partition/procedure and explicit agreement about the
   upstream build's otherwise unspecified default loop scheduling.
5. First insertion assigns the product; repeated insertion invokes
   Add(product,accumulator). Record any semiring-contract adaptation explicitly.
6. The same row sorting and direct native tuple emission, which already match.

Checked allocation, representable unsigned hash arithmetic, race-free thread
count discovery and move-only ownership are SpCraft safety/ownership
adaptations. They should be separately disclosed and proved equivalent on the
shared valid range, not silently presented as identical source. There is no
reason to reproduce signed-overflow undefined behavior or a data race to
establish a faithful algorithmic baseline.

No production kernel was changed during this mapping review. The differences
above describe the actual implementation, including the dense lookup and
checked block-prefix changes made before this review. The existing
[correctness audit](ColumnCorrectnessAudit.md) addresses SpCraft's own
invariants; it is not evidence that CombBLAS executes identical logic.

## Reviewed source fingerprints

```text
e9ef1220de878e5cc55ce7385c9499865915f2a9bd82a8ed0185b797df15c45c  include/kernel/mtSpGEMMColumn.h
57313147b70bbbdd6c9facb04224043747d01e31dcedbbc540d1314adb92cf46  include/kernel/SpGEMMHash.h
281645c64f6b0bd09ddc44248b2c00ff6215250ab908991880c68bd0ac2b7da5  include/core/TupleMatrix.h
edfa0c39431be6508115ee7913ba5aba61bfd74937fafb0ff9aa5c76a4e29b4c  build/_deps/combblas-src/include/CombBLAS/mtSpGEMM.h
03990005ee5cfb2aa381f4ad090ef590ff15e2f2f2285bc62d9bdfd811c9454d  build/_deps/combblas-src/include/CombBLAS/dcsc.cpp
```
