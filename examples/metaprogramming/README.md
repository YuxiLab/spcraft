# C++ Metaprogramming in High-Performance Sparse Computation

Welcome to the **Metaprogramming Guide** for `SPCraft` (`iusparse`). This directory documents the modern C++ (C++20) template metaprogramming techniques, zero-cost abstractions, and compile-time safety mechanisms engineered for high-performance sparse matrix computation.

---

## Table of Contents
1. [Overview & Motivation](#1-overview--motivation)
2. [3-Tier Type Parameterization (`IT`, `NT`, `OT`)](#2-3-tier-type-parameterization-it-nt-ot)
3. [Resource Ownership, RAII & Move Semantics](#3-resource-ownership-raii--move-semantics)
4. [Compile-Time Diagnostics & C++20 Concepts](#4-compile-time-diagnostics--c20-concepts)
5. [Branch Elimination via `if constexpr`](#5-branch-elimination-via-if-constexpr)
6. [Expression Templates & Kernel Fusion](#6-expression-templates--kernel-fusion)
7. [Policy-Based Design](#7-policy-based-design)
8. [Building & Running the Examples](#8-building--running-the-examples)

---

## 1. Overview & Motivation

Sparse matrix operations (such as Sparse Matrix-Vector Multiplication, **SpMV**) are fundamentally **memory-bandwidth bound**. The computational intensity (FLOPs per byte transferred) is very low:
$$\text{Arithmetic Intensity} \approx \frac{2 \cdot \text{nnz}}{(\text{sizeof}(OT) \cdot m) + (\text{sizeof}(IT) + \text{sizeof}(NT)) \cdot \text{nnz}} \ll 1$$

In traditional object-oriented architectures, abstractions like inheritance, dynamic polymorphism (virtual function tables), and generic object pointers (`void*`) introduce:
1. **Dynamic dispatch overhead**: Virtual function indirection prevents inlining and CPU branch prediction.
2. **Data bloat**: Object headers and alignment padding inflate cache footprints.
3. **Implicit runtime copies**: Passing large sparse data structures by value creates multi-gigabyte memory allocations.

**Template Metaprogramming (TMP)** in modern C++ shifts these costs from runtime to compile time, delivering the **Zero-Overhead Principle**: *What you don't use, you don't pay for. And what you do use, you couldn't hand code any better.*

```
                 C++ Template Metaprogramming in SPCraft
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   1. Decoupled 3-Tier Types      ──► Saves up to 50% Memory Bandwidth   │
  │      CsrMatrix<IT, NT, OT>                                              │
  │                                                                         │
  │   2. Move-Only RAII Semantics   ──► Zero Copy Overhead, No Double-Free  │
  │      (= delete copy ctor)                                               │
  │                                                                         │
  │   3. C++20 Concepts & Traits     ──► Clean Compile-Time Error Messages  │
  │      concept SparseValue                                                │
  │                                                                         │
  │   4. `if constexpr` Dispatch    ──► Zero Runtime Branch Penalties      │
  │      Static Kernel Selection                                            │
  │                                                                         │
  │   5. Expression Templates        ──► Single-Pass Fused Kernel Loops     │
  │      Lazy AST Evaluation                                                │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 3-Tier Type Parameterization (`IT`, `NT`, `OT`)

### The Challenge
In real-world graphs (e.g. social networks, web crawls, genomics), a sparse matrix can have $m = 50,000,000$ vertices and $\text{nnz} = 3,000,000,000$ edges.
- The row/column indices ($50 \times 10^6$) easily fit in a standard 32-bit signed integer (`int32_t`, max $2.14 \times 10^9$).
- The nonzero count ($3 \times 10^9$) exceeds `int32_t` and requires a 64-bit offset (`int64_t`).

If a sparse library forces a single index type across the entire data structure:
- **Using `int32_t`**: Overflow crash when indexing beyond 2 billion nonzeros.
- **Using `int64_t`**: Inflates `col_id` from 4 bytes to 8 bytes per nonzero. For 3 billion entries, this wastes **12 GB of DRAM bandwidth** on every SpMV iteration!

### The Solution in SPCraft
SPCraft cleanly separates the types into a 3-tier generic template:

```cpp
namespace spcraft {
template <class IT, class NT, class OT = IT>
class CsrMatrix {
  OT* row_ptr = nullptr;  // Size: m + 1  (Offset Type: int64_t)
  IT* col_id  = nullptr;  // Size: nnz    (Index Type:  int32_t)
  NT* val     = nullptr;  // Size: nnz    (Numeric Type: float/double)
  ...
};
}
```

See [01_type_parameterization.cpp](01_type_parameterization.cpp) for the memory footprint analysis.

---

## 3. Resource Ownership, RAII & Move Semantics

### Deleting Copy Operations
To protect large-scale scientific applications from silent performance degradation, `CsrMatrix` and `CooMatrix` explicitly delete the copy constructor and copy assignment operator:

```cpp
CsrMatrix(const CsrMatrix<IT, NT, OT>& rhs) = delete;
CsrMatrix& operator=(const CsrMatrix<IT, NT, OT>& rhs) = delete;
```

### Explaining `e02_compiler_error.cpp`
When a user writes:
```cpp
CsrMatrix<int32_t, float> A;
CsrMatrix<int32_t, float> B = A; // ERROR!
```
The C++ compiler halts compilation immediately with:
```
error: call to deleted constructor of 'spcraft::CsrMatrix<int32_t, float>'
```
This forces the developer to choose:
1. **Transfer Ownership** with zero cost: `CSR B = std::move(A);`
2. **Explicit Deep Copy**: `CSR B = A.Clone();`
3. **Non-Owning View**: Wrap externally managed pointers (`memowned = false`).

See [02_ownership_and_move_semantics.cpp](02_ownership_and_move_semantics.cpp).

---

## 4. Compile-Time Diagnostics & C++20 Concepts

Legacy C++ templates produced notorious, deeply nested compiler error dumps when an inappropriate type was passed. Modern C++20 **Concepts** allow expressing type constraints directly in the function/class signature:

```cpp
template <typename T>
concept SparseIndex = std::integral<T> && (sizeof(T) >= 2);

template <typename T>
concept SparseValue = std::floating_point<T> || std::integral<T>;

template <SparseIndex IT, SparseValue NT, SparseOffset OT>
void SpMV(const CsrMatrix<IT, NT, OT>& A, const NT* x, NT* y);
```

If an unsupported type (such as `std::string`) is supplied, the compiler produces a concise 1-line failure:
```
error: constraints not satisfied for class template 'ValidatedCsrMatrix'
note: because 'std::string' does not satisfy 'SparseValue'
```

See [03_concepts_and_constraints.cpp](03_concepts_and_constraints.cpp).

---

## 5. Branch Elimination via `if constexpr`

In inner SpMV loops, runtime `if` conditions (e.g. checking matrix symmetry, zero elements, or numeric formats) cause branch mispredictions and prevent SIMD vectorization.

Using `if constexpr`, compiler branches are evaluated during template instantiation. The unselected branches are completely discarded before assembly generation:

```cpp
template <MatrixSymmetry Symmetry, typename IT, typename NT, typename OT>
void FastSpMV(const CsrMatrix<IT, NT, OT>& A, const NT* x, NT* y) {
  if constexpr (Symmetry == MatrixSymmetry::General) {
    // Generates tight vectorizable loop without symmetry checks
  } else if constexpr (Symmetry == MatrixSymmetry::Symmetric) {
    // Generates symmetric update logic
  }
}
```

See [04_constexpr_dispatch.cpp](04_constexpr_dispatch.cpp).

---

## 6. Expression Templates & Kernel Fusion

Consider evaluating the vector update:
$$y = \alpha \cdot (A \cdot x) + \beta \cdot z$$

In naive C++, overloaded operators `*` and `+` construct intermediate temporary vectors, causing multiple memory allocation calls and cache evictions.

**Expression Templates** construct a lightweight Abstract Syntax Tree (AST) at compile time. When assigned to the output vector `y`, the entire expression is evaluated in a single streaming loop:

```cpp
// Builds AST without allocating memory:
auto expr = (alpha * (A * x)) + (beta * z);

// Evaluates in a single pass directly into y:
y = expr;
```

See [05_expression_templates.cpp](05_expression_templates.cpp).

---

## 7. Policy-Based Design

Policy-based design decomposes complex classes into orthogonal compile-time policies. For sparse computation, we can customize:
- **Storage Policy**: `OwningStoragePolicy` vs `ViewStoragePolicy`.
- **Indexing Policy**: `ZeroBasedPolicy` (C/C++) vs `OneBasedPolicy` (Fortran/MATLAB/MatrixMarket).

```cpp
using FortranCsr = PolicySparseMatrix<
    int32_t, double, int32_t,
    OwningStoragePolicy,
    OneBasedPolicy>;
```

All index translation functions (`idx - 1`) are fully inlined with zero runtime overhead.

See [06_policy_based_design.cpp](06_policy_based_design.cpp).

---

## 8. Building & Running the Examples

All examples are built through CMake:

```bash
# 1. Configure the project
cmake -B build -S . -DSPCRAFT_BUILD_EXAMPLES=ON

# 2. Build the metaprogramming examples
cmake --build build --target 01_type_parameterization \
                           02_ownership_and_move_semantics \
                           03_concepts_and_constraints \
                           04_constexpr_dispatch \
                           05_expression_templates \
                           06_policy_based_design

# 3. Run any example
./build/examples/metaprogramming/01_type_parameterization
./build/examples/metaprogramming/02_ownership_and_move_semantics
./build/examples/metaprogramming/03_concepts_and_constraints
./build/examples/metaprogramming/04_constexpr_dispatch
./build/examples/metaprogramming/05_expression_templates
./build/examples/metaprogramming/06_policy_based_design
```
