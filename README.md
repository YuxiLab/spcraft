# Sparse Craft Library

This is a teaching-oriented sparse computation library. It aims to provide a light-weight but clear-documented sparse kernel, sparse algorithm library with visualization and teaching tutorials.

At the ground layer, the library will focus on implementing the basic sparse matrix storage, sparse matrix kernel ( SpMV, SpMM, etc.) as well as reproducing some classic papers in the sparse computation community.

Please send email to `yuxihong@iu.edu` if you have any questions.

[Yuxi Hong](https://hongyx11.github.io/)

## CUDA code navigation on macOS

Run `python3 scripts/setup_macos_clangd.py` to prepare local clangd parsing for
C++ and CUDA without installing a GPU driver or building the library. See
[the setup guide](scripts/clangd/README.md) for verification and editor details.

## Private lecture notes

The optional `latex/` submodule contains private lecture notes and is not needed
to build or use SPCraft. Authorized collaborators can check it out with:

```bash
git submodule update --init latex
```


## Design Philosophy

A lot of class design of Sparse Craft Library is derived from [CombBLAS](https://github.com/PASSIONLab/CombBLAS).
Thank you Aydin!

## Sparse Matrix Storage

`CooMatrix<IT, NT, OT = IT>` stores one contiguous raw array:

```cpp
using COO = spcraft::CooMatrix<int, double>;
using Entry = COO::Entry;  // std::tuple<int, int, double>: row, column, value
COO matrix;
matrix.Allocate(2, 3, 4);
Entry* entries = matrix.entries;
entries[0] = {2, 1, 7.0};
entries[1] = {0, 3, -5.0};
COO view(entries, matrix.nnz, matrix.m, matrix.n); // borrows live entries
const auto csr = matrix.ToCsr();
```

COO owns allocation and entry lifetimes through `malloc/free`, explicit tuple
construction/destruction, and `memowned`. It is move-only; `Clone()` makes an
independent copy. Allocation builds the replacement before releasing old storage,
so failure preserves the owner. `Reset()` drops ownership without freeing and is
intended for moved-from objects or views. Empty matrices retain their shape.
The normal array allocation value-initializes every tuple component.

General COO permits arbitrary entry order and retains duplicates and explicit
zeros. `ToCsr()` groups by row while preserving within-row order and validates
coordinates. Matrix Market parsing uses temporary component vectors, then copies
into the owned `Entry*`; no vector backs the matrix. Python arrays are copied at
both boundaries, including when extracting a component of an entry array.

The native-output `OmpHashSpGEMM` returns this same COO owner directly, with
entries sorted by column then row. The former separate tuple container and
COO's three-array constructor/fields are removed: use `entries[p]` or structured
bindings instead of COO `row_id[p]`, `col_id[p]`, and `val[p]`. CSR, CSC and DCSC
retain their compressed arrays. See the [COO example](examples/spcraft/e01_CooMatrix.cpp).

## Sparse Kernels Implementation

`OmpHashSpGEMM<SemiRing>(A, B)` in [mtSpGEMM.h](include/kernel/mtSpGEMM.h) is the
single maintained OpenMP sparse-sparse multiplication implementation. It maps
CombBLAS's column lookup, full work estimation, symbolic hash counting, prefixes,
numeric hash accumulation and sorted tuple emission. No CombBLAS dependency is
needed to use the library kernel.

```cpp
using Ring = spcraft::PlusTimesRing<double>;
// A and B are canonical DcscMatrix<int, double>, with A.n == B.m.
auto C = spcraft::OmpHashSpGEMM<Ring>(A, B);  // DCSC -> CooMatrix
```

Inputs must have sorted unique stored column IDs and sorted unique row IDs in
each stored column. `A.n + 1` must fit IT and be at most 2^24, and full work and
output counts must fit OT. Output retains structural zeros. Convert CSC inputs
explicitly with `DcscMatrix::FromCsc`; that conversion preserves row order and
duplicates, so callers must supply canonical CSC. Convert the COO result with
`ToCsr()` when a consumer needs CSR.

The old CSR/CSC/DCSC compressed-output kernels and alternate tuple kernel are
removed. There are no compatibility entry points or hidden kernel-selection
flags. The [example](examples/spcraft/e03_column_spgemm.cpp),
[source correspondence](benchmarks/combblas/SourceMapping.md), and
[kernel tests](tests/test_mtspgemm.cpp) describe the supported contract.

Python's CSR `spgemm` method explicitly sorts and merges duplicate input
coordinates into canonical DCSC, calls this same kernel, then converts COO to
CSR. These boundary conversions cost additional time and are excluded from the
[native comparison benchmark](benchmarks/combblas/README.md).


## Optional Python interface

Passing `-DSPCRAFT_BUILD_PYTHON=ON` to cmake to activate python interface. Please use [uv](https://docs.astral.sh/uv/) to install the python environment.

to install `uv`, use
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use `uv` to create a virutal environment and install the python interface.
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .
```

Create COO or CSR matrices from NumPy-compatible arrays:

```python
import numpy as np
import spcraft

coo = spcraft.coo_matrix(
    row_indices=[0, 1, 1],
    column_indices=[1, 0, 2],
    values=np.array([2, 4, 3], dtype=np.float32),
    shape=(2, 3),
)
csr = coo.to_csr()

print(csr.row_offsets)
print(csr.column_indices)
print(csr.values)
```

Generate graph inputs directly with SPCraft's C++ implementations:

```python
er = spcraft.gen_er_graph(vertices=512, expected_degree=12, seed=2026)
rmat = spcraft.gen_rmat(scale=9, edge_factor=6, seed=2026)
```

Inputs and returned NumPy arrays are copied in this first implementation. This keeps ownership
unambiguous: modifying a Python array cannot invalidate or modify an SPCraft matrix.
