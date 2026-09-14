# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (torchgpu)
#     language: python
#     name: torchgpu
# ---

# %% [markdown]
# # Chapter 1. Sparse Matrix Format
# ## Section 1. Coordinate Format
#
# The **Coordinate (COO)** format is the most intuitive way to store a sparse matrix. Instead of storing a full 2D grid, we only record the coordinates (row, column) and the value of each non-zero element.
#
# This format is represented using **three flat arrays** of the same length ($nnz$):
# * **`row`**: The row index of each nonzero element.
# * **`col`**: The column index of each nonzero element.
# * **`data`**: The actual numerical value at those coordinates.
#
# ### Why use COO?
# 1. **Very easy to construct**: It's incredibly simple to build an append-only list of triplets as data is generated.
# 2. **Space-efficient for storage**: Unlike a dense matrix, its memory usage is strictly proportional to the number of nonzeros ($O(nnz)$), completely independent of the matrix dimensions ($n$).
#
# ### The Downside
# * **memory consumption**: But for denser sparse matrix, the $nnz$ in the same column / row will share the same column / row indices, but COO will still duplicate for each of the points.
# * **No fast slicing**: If you want to grab an entire column (like $A(:, j)$) or query a single entry $A(i, j)$, you must search through the entire list of coordinates. This takes $O(nnz)$ linear time.
#

# %%
import numpy as np
import scipy.sparse as sp

# Nonzeros: (5, 0) = 0.1, (7, 0) = 0.2, (3, 6) = 0.3, (1, 7) = 0.4
# Coordinate Format Input
rows = np.array([5,   7,   3,   1  ])
cols = np.array([0,   0,   6,   7  ])
vals = np.array([0.1, 0.2, 0.3, 0.4])

# 2. Construct SciPy's native COO matrix
A_coo = sp.coo_matrix((vals, (rows, cols)), shape=(9, 9))

print("=== SciPy COO Matrix ===")
print(A_coo)

# 3. Create a standard SciPy CSC matrix using COO format input
A_csc = sp.csc_matrix((vals, (rows, cols)), shape=(9, 9))

print("=== Standard CSC Arrays ===")
print("NUM (Values):     ",  A_csc.data   )
print("IR (Row Indices):  ", A_csc.indices)
print("JC (Col Pointers): ", A_csc.indptr )

# %% 
# ## 
import numpy as np

# %% [markdown]
# # Quiz

# %% [markdown]
# [📄 Open PDF in VS Code](../latex/figures/obj/dcsc_fig2_csc.pdf)
#

# %%
