# SPCraft Python examples

These small programs connect one sparse matrix-vector multiplication to the
algorithms built around it. Run them from the repository root after installing
the optional Python interface with `uv sync --all-extras`.

```bash
uv run python python/examples/e01_spmv.py --output spmv_example.png
uv run python python/examples/e02_conjugate_gradient.py --output cg_solution.png
uv run python python/examples/e03_pagerank.py --output pagerank_graph.png
```

- `e01_spmv.py` compares SPCraft with SciPy and plots the CSR pattern, input,
  and output vector.
- `e02_conjugate_gradient.py` solves a five-point 2D Poisson problem whose exact
  solution is known, then plots the exact field, computed field, and error.
- `e03_pagerank.py` includes a dangling page and plots both the directed graph
  and its converged rank distribution.

Use `--help` on any program to see its matrix size, solver controls, and output
filename options. The Matplotlib `Agg` backend makes every example runnable on
headless compute nodes.
