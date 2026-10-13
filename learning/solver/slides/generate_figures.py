"""Generate editable TikZ figures and metrics from the teaching solver."""

from pathlib import Path
from math import sin
import sys

SLIDES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SLIDES_DIR.parent))

from sparse_solver import CSR, poisson_2d
from superlu_pipeline import factorize


def entries(A):
    return {(i, A.indices[k]) for i in range(A.shape[0])
            for k in range(A.indptr[i], A.indptr[i + 1])}


def factor_pattern(solver):
    positions = {(i, j) for i, row in enumerate(solver.L)
                 for j, value in row.items() if value != 0}
    positions.update((i, j) for i, row in enumerate(solver.U) for j in row)
    return positions


def permuted_input_pattern(solver):
    inverse_rows = {old: new for new, old in enumerate(solver.row_order)}
    inverse_columns = {old: new for new, old in enumerate(solver.analysis.col_order)}
    return {(inverse_rows[i], inverse_columns[j]) for i, j in entries(solver.A)}


def matrix_plot(path, n, positions, original, width=4.0):
    unit = width / n
    lines = [f"\\begin{{tikzpicture}}[x={unit:.6f}cm,y={unit:.6f}cm]",
             f"\\fill[Mist] (0,0) rectangle ({n},-{n});"]
    for i, j in sorted(positions):
        color = "Teal" if (i, j) in original else "Amber"
        lines.append(f"\\fill[{color}] ({j + .12:.3f},-{i + .12:.3f}) "
                     f"rectangle ({j + .88:.3f},-{i + .88:.3f});")
    lines.append(f"\\draw[Slate!35] (0,0) rectangle ({n},-{n});")
    if n <= 16:
        lines.append(f"\\draw[Slate!15,step=1] (0,0) grid ({n},-{n});")
    lines.extend([
        f"\\node[anchor=south,font=\\scriptsize] at (.5,0) {{0}};",
        f"\\node[anchor=south,font=\\scriptsize] at ({n - .5},0) {{{n - 1}}};",
        f"\\node[anchor=east,font=\\scriptsize] at (0,-.5) {{0}};",
        f"\\node[anchor=east,font=\\scriptsize] at (0,-{n - .5}) {{{n - 1}}};",
        "\\end{tikzpicture}",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    figures = SLIDES_DIR / "figures"
    figures.mkdir(exist_ok=True)
    small = poisson_2d(4)
    small_solver = factorize(small, ordering="natural", postorder=False)
    matrix_plot(figures / "title_input.tex", 16, entries(small), entries(small), 3.0)
    matrix_plot(figures / "title_factors.tex", 16, factor_pattern(small_solver),
                permuted_input_pattern(small_solver), 3.0)

    A = poisson_2d(8)
    natural = factorize(A, ordering="natural", postorder=False)
    ordered = factorize(A, ordering="minimum_degree")
    matrix_plot(figures / "poisson_input.tex", 64, entries(A), entries(A), 3.4)
    matrix_plot(figures / "poisson_natural.tex", 64, factor_pattern(natural),
                permuted_input_pattern(natural), 3.4)
    matrix_plot(figures / "poisson_ordered.tex", 64, factor_pattern(ordered),
                permuted_input_pattern(ordered), 3.4)

    n = A.shape[0]
    r = [10 ** (6 * i / (n - 1)) for i in range(n)]
    c = [10 ** (-4 * i / (n - 1)) for i in range(n)]
    scaled = CSR(A.shape, tuple(A.data[k] * r[i] * c[A.indices[k]]
                               for i in range(n)
                               for k in range(A.indptr[i], A.indptr[i + 1])),
                 A.indices, A.indptr)
    expected = [1 + sin(.13 * i) for i in range(n)]
    report = factorize(scaled).refine(scaled.matvec(expected), [x + .001 for x in expected])
    coordinates = " ".join(f"({i},{max(value, 1e-19):.12e})"
                           for i, value in enumerate(report.backward_error_history))
    xmax = max(1.5, len(report.backward_error_history) - .5)
    xticks = ",".join(str(i) for i in range(len(report.backward_error_history)))
    refinement_plot = (r"""\begin{tikzpicture}
\begin{semilogyaxis}[
 width=6.4cm,height=4.6cm,
 xlabel={Correction steps},ylabel={Componentwise backward error},
 xmin=-0.15,xmax=__XMAX__,xtick={__XTICKS__},
 ymin=1e-18,ymax=1e-2,
 grid=major,grid style={Slate!15},
 tick label style={font=\scriptsize},label style={font=\small},
 axis line style={Slate!45},
]
\addplot[Teal,very thick,mark=*,mark size=2.5pt] coordinates {"""
        + coordinates + r"""};
\addplot[Amber,dashed,thick,domain=-0.15:__XMAX__] {2.220446049250313e-16};
\node[anchor=south east,font=\scriptsize,text=Amber!80!black]
 at (axis cs:__LABELX__,2.220446049250313e-16) {machine epsilon};
\end{semilogyaxis}
\end{tikzpicture}
""")
    (figures / "refinement.tex").write_text(
        refinement_plot.replace("__XMAX__", str(xmax)).replace("__XTICKS__", xticks)
        .replace("__LABELX__", str(xmax - .02))
    )
    metrics = {
        "NaturalLNNZ": natural.nnz_L,
        "NaturalUNNZ": natural.nnz_U,
        "OrderedLNNZ": ordered.nnz_L,
        "OrderedUNNZ": ordered.nnz_U,
        "InputNNZ": len(A.data),
        "NaturalCombinedNNZ": natural.nnz_L + natural.nnz_U - n,
        "OrderedCombinedNNZ": ordered.nnz_L + ordered.nnz_U - n,
        "RefinementSteps": report.refinement_steps,
    }
    (figures / "metrics.tex").write_text("\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in metrics.items()) + "\n")
    print(f"Generated 6 TikZ figures; natural/ordered combined nnz: "
          f"{metrics['NaturalCombinedNNZ']}/{metrics['OrderedCombinedNNZ']}")
    print(f"Refinement backward-error history: {report.backward_error_history}")


if __name__ == "__main__":
    main()
