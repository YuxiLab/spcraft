# Eigen dense-solver lecture examples

These small programs accompany the six Beamer decks named
`dense_solvers_01_*.tex` through `dense_solvers_06_*.tex` in `latex/beamer/`.
Each program teaches one numerical idea and accepts parameters through
`cxxopts`.

| Program | Main idea | Lecture |
| --- | --- | --- |
| `e01_triangular_solve` | Forward and backward substitution | 1 |
| `e02_lu_pivoting` | Failure without pivoting | 2 |
| `e03_conditioning_refinement` | Residual, forward error, and refinement | 3 |
| `e04_cholesky` | Cholesky versus pivoted LU | 4 |
| `e05_least_squares` | Normal equations versus pivoted QR | 5 |
| `e06_rank_deficient_svd` | Numerical rank and minimum-norm solution | 5 |
| `e07_dense_vs_sparse` | Dense and sparse direct solution | 6 |
| `e08_conjugate_gradient` | CG convergence history in CSV form | 6 |

Build from the SPCraft repository root:

```bash
cmake -S . -B build -DSPCRAFT_BUILD_EXAMPLES=ON
cmake --build build -j
```

Every executable supports `--help`. For example:

```bash
build/bin/e02_lu_pivoting --pivot 1e-20
build/bin/e05_least_squares --points 30 --degree 12
build/bin/e08_conjugate_gradient --grid 32 > cg-history.csv
```
