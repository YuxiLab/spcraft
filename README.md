# SPCraft

This is a teaching-oriented sparse computation library.
It aims to provide a light-weight but clear-documented sparse kernel, sparse algorithm library with visualization and teaching tutorials.

At the ground layer, the library will focus on implementing the basic sparse matrix storage, sparse matrix kernel ( SpMV, SpMM, etc.) as well as reproducing some classic papers in the sparse computation community.

Please send email to yuxihong@iu.edu if you have any questions.

Yuxi Hong

## Optional Python interface

The Python package is a thin, optional nanobind interface. The C++ library does not depend on
Python or nanobind, and the `SPCRAFT_BUILD_PYTHON` CMake option is disabled by default.

Build and install the package from the repository root:

```bash
python -m pip install .
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

Inputs and returned NumPy arrays are copied in this first implementation. This keeps ownership
unambiguous: modifying a Python array cannot invalidate or modify an SPCraft matrix.
