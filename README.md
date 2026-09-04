# Sparse Craft Library

This is a teaching-oriented sparse computation library. It aims to provide a light-weight but clear-documented sparse kernel, sparse algorithm library with visualization and teaching tutorials.

At the ground layer, the library will focus on implementing the basic sparse matrix storage, sparse matrix kernel ( SpMV, SpMM, etc.) as well as reproducing some classic papers in the sparse computation community.

Please send email to `yuxihong@iu.edu` if you have any questions.

[Yuxi Hong](https://hongyx11.github.io/)


## Design Philosophy

A lot of class design of Sparse Craft Library is derived from [CombBLAS](https://github.com/PASSIONLab/CombBLAS).
Thank you Aydin!

## Sparse Matrix Storage



## Sparse Kernels Implementation



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
