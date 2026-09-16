#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <string>

#include "SpCraft.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

using Index = int32_t;
using Offset = int32_t;

template <class T>
using InputArray = nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using OutputArray = nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object CopyToNumpy(const T* source, size_t size)
{
  T* copy = new T[size];
  if (size != 0) {
    std::copy(source, source + size, copy);
  }

  nb::capsule owner(copy, [](void* pointer) noexcept { delete[] static_cast<T*>(pointer); });
  OutputArray<T> array(copy, {size}, owner);
  return nb::cast(array, nb::rv_policy::reference);
}

//! Hand an already-allocated buffer to NumPy, transferring ownership.
template <class T>
nb::object AdoptAsNumpy(T* data, size_t size)
{
  nb::capsule owner(data, [](void* pointer) noexcept { delete[] static_cast<T*>(pointer); });
  OutputArray<T> array(data, {size}, owner);
  return nb::cast(array, nb::rv_policy::reference);
}

void ValidateShape(Index rows, Index columns)
{
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("matrix dimensions must be non-negative");
  }
}

Offset CheckedNnz(size_t size)
{
  if (size > static_cast<size_t>(std::numeric_limits<Offset>::max())) {
    throw std::overflow_error("number of nonzeros exceeds the int32 offset range");
  }
  return static_cast<Offset>(size);
}

void ValidateColumnIndices(InputArray<Index> column_indices, Index columns)
{
  for (size_t i = 0; i < column_indices.size(); ++i) {
    const Index column = column_indices.data()[i];
    if (column < 0 || column >= columns) {
      throw std::invalid_argument("column index is outside the matrix shape");
    }
  }
}

template <class Number>
spcraft::CooMatrix<Index, Number, Offset> CooFromArrays(InputArray<Index> row_indices,
                                                        InputArray<Index> column_indices,
                                                        InputArray<Number> values, Index rows,
                                                        Index columns)
{
  ValidateShape(rows, columns);
  if (row_indices.size() != column_indices.size() || row_indices.size() != values.size()) {
    throw std::invalid_argument("COO row, column, and value arrays must have equal lengths");
  }

  ValidateColumnIndices(column_indices, columns);
  for (size_t i = 0; i < row_indices.size(); ++i) {
    const Index row = row_indices.data()[i];
    if (row < 0 || row >= rows) {
      throw std::invalid_argument("row index is outside the matrix shape");
    }
  }

  const Offset nnz = CheckedNnz(values.size());
  spcraft::CooMatrix<Index, Number, Offset> result;
  result.Allocate(nnz, rows, columns);
  for (Offset i = 0; i < nnz; ++i)
    result.entries[i] = {row_indices.data()[i], column_indices.data()[i], values.data()[i]};
  return result;
}

template <class Number>
spcraft::CsrMatrix<Index, Number, Offset> CsrFromArrays(InputArray<Offset> row_offsets,
                                                        InputArray<Index> column_indices,
                                                        InputArray<Number> values, Index rows,
                                                        Index columns)
{
  ValidateShape(rows, columns);
  const size_t expected_offsets = static_cast<size_t>(rows) + 1;
  if (row_offsets.size() != expected_offsets) {
    throw std::invalid_argument("CSR row_offsets length must equal rows + 1");
  }
  if (column_indices.size() != values.size()) {
    throw std::invalid_argument("CSR column and value arrays must have equal lengths");
  }

  const Offset nnz = CheckedNnz(values.size());
  if (row_offsets.data()[0] != 0 || row_offsets.data()[rows] != nnz) {
    throw std::invalid_argument("CSR row_offsets must start at 0 and end at nnz");
  }
  for (Index row = 0; row < rows; ++row) {
    const Offset begin = row_offsets.data()[row];
    const Offset end = row_offsets.data()[row + 1];
    if (begin < 0 || begin > end || end > nnz) {
      throw std::invalid_argument("CSR row_offsets must be monotonic and within [0, nnz]");
    }
  }
  ValidateColumnIndices(column_indices, columns);

  spcraft::CsrMatrix<Index, Number, Offset> result;
  result.Allocate(nnz, rows, columns);
  std::copy(row_offsets.data(), row_offsets.data() + expected_offsets, result.row_ptr);
  if (nnz != 0) {
    std::copy(column_indices.data(), column_indices.data() + nnz, result.col_id);
    std::copy(values.data(), values.data() + nnz, result.val);
  }
  return result;
}

template <class Number>
spcraft::CsrMatrix<Index, Number, Offset> GenerateERGraph(Index vertices, double expected_degree,
                                                          std::uint64_t seed)
{
  return spcraft::GenERGraph<Number, Index, Offset>(vertices, expected_degree, seed);
}

template <class Number>
spcraft::CsrMatrix<Index, Number, Offset> GenerateRMAT(Index scale, std::size_t edge_factor,
                                                       std::uint64_t seed, double a, double b,
                                                       double c, double d)
{
  return spcraft::GenRMAT<Number, Index, Offset>(scale, edge_factor, seed, a, b, c, d);
}

//! y = A x, returned as a fresh NumPy array. Shared by spmv, dot and @.
template <class Number>
nb::object SpmvToNumpy(const spcraft::CsrMatrix<Index, Number, Offset>& matrix,
                       InputArray<Number> x)
{
  if (x.size() != static_cast<size_t>(matrix.n)) {
    throw std::invalid_argument("vector length must match matrix column dimension");
  }
  Number* y = new Number[matrix.m];
  // Both views borrow: SpCraft must not free NumPy's buffer or the capsule's.
  const spcraft::DenseVector<Index, Number> input(const_cast<Number*>(x.data()),
                                                  static_cast<Index>(x.size()));
  spcraft::DenseVector<Index, Number> output(y, matrix.m);
  spcraft::OmpSpMV<spcraft::PlusTimesRing<Number>>(matrix, input, output);
  return AdoptAsNumpy(y, static_cast<size_t>(matrix.m));
}

//! Python accepts arbitrary CSR assembly; canonicalize it at this format boundary.
//! This prepares inputs only. OmpHashSpGEMM remains the sole multiplication kernel.
template <class Number>
spcraft::DcscMatrix<Index, Number, Offset> CanonicalDcscFromCsr(
    const spcraft::CsrMatrix<Index, Number, Offset>& matrix)
{
  using Ring = spcraft::PlusTimesRing<Number>;
  spcraft::CooMatrix<Index, Number, Offset> coordinates;
  coordinates.Allocate(matrix.nnz, matrix.m, matrix.n);
  for (Index row = 0; row < matrix.m; ++row)
    for (Offset p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p)
      coordinates.entries[p] = {row, matrix.col_id[p], matrix.val[p]};
  if (matrix.nnz > 1)
    std::stable_sort(coordinates.entries, coordinates.entries + matrix.nnz,
                     [](const auto& a, const auto& b) {
                       return std::tie(a.col, a.row) < std::tie(b.col, b.row);
                     });
  Offset nonzeros = 0;
  Index columns = 0;
  for (Offset p = 0; p < matrix.nnz; ++p) {
    const auto& entry = coordinates.entries[p];
    if (nonzeros && coordinates.entries[nonzeros - 1].row == entry.row &&
        coordinates.entries[nonzeros - 1].col == entry.col) {
      auto& value = coordinates.entries[nonzeros - 1].val;
      value = Ring::Add(value, entry.val);
    } else {
      if (!nonzeros || coordinates.entries[nonzeros - 1].col != entry.col) ++columns;
      coordinates.entries[nonzeros++] = entry;
    }
  }
  spcraft::DcscMatrix<Index, Number, Offset> result;
  result.Allocate(nonzeros, matrix.m, matrix.n, columns);
  Index column = 0;
  for (Offset p = 0; p < nonzeros; ++p) {
    const auto& [row, col, value] = coordinates.entries[p];
    if (!p || col != coordinates.entries[p - 1].col) {
      result.col_id[column] = col;
      result.col_ptr[column++] = p;
    }
    result.row_id[p] = row;
    result.val[p] = value;
  }
  result.col_ptr[columns] = nonzeros;
  return result;
}

template <class Number>
void BindCsr(nb::module_& module, const char* name)
{
  using Matrix = spcraft::CsrMatrix<Index, Number, Offset>;

  nb::class_<Matrix>(module, name)
      .def_static("from_arrays", &CsrFromArrays<Number>, "row_offsets"_a, "column_indices"_a,
                  "values"_a, "rows"_a, "columns"_a,
                  "Create a matrix by copying one-dimensional NumPy arrays.")
      .def("clone", &Matrix::Clone)
      .def_prop_ro("shape", [](const Matrix& matrix) { return nb::make_tuple(matrix.m, matrix.n); })
      .def_ro("nnz", &Matrix::nnz)
      .def_prop_ro("row_offsets",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.row_ptr, static_cast<size_t>(matrix.m) + 1);
                   })
      .def_prop_ro("column_indices",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.col_id, static_cast<size_t>(matrix.nnz));
                   })
      .def_prop_ro("values",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.val, static_cast<size_t>(matrix.nnz));
                   })
      .def("spmv", &SpmvToNumpy<Number>, "x"_a, "Multiply the sparse matrix by a dense vector x.")
      .def("__matmul__", &SpmvToNumpy<Number>, "x"_a)
      .def("dot", &SpmvToNumpy<Number>, "x"_a)
      .def(
          "spgemm",
          [](const Matrix& matrix, const Matrix& other) {
            if (matrix.n != other.m)
              throw std::invalid_argument("SpGEMM matrix dimensions do not match");
            const auto a = CanonicalDcscFromCsr(matrix), b = CanonicalDcscFromCsr(other);
            return spcraft::OmpHashSpGEMM<spcraft::PlusTimesRing<Number>>(a, b).ToCsr();
          },
          "other"_a, "Convert inputs to canonical DCSC, call OmpHashSpGEMM, and return CSR.")
      .def(
          "pagerank",
          [](const Matrix& matrix, Number damping, Number tolerance, int max_iterations) {
            // The adjacency matrix is transposed and normalized once here; the
            // iteration itself is a sequence of SpMVs against the result.
            const Matrix op = spcraft::pagerank_operator(matrix);
            spcraft::PageRankOptions<Number> options;
            options.damping = damping;
            options.tolerance = tolerance;
            options.max_iterations = max_iterations;

            Number* ranks = new Number[matrix.m];
            spcraft::DenseVector<Index, Number> rank(ranks, matrix.m);
            const auto result = spcraft::pagerank_openmp(op, rank, options);

            nb::dict info;
            info["iterations"] = result.iterations;
            info["residual"] = result.residual;
            info["converged"] = result.converged;
            return nb::make_tuple(AdoptAsNumpy(ranks, static_cast<size_t>(matrix.m)), info);
          },
          "damping"_a = Number{0.85}, "tolerance"_a = Number{1e-10}, "max_iterations"_a = 100,
          "PageRank of the graph whose adjacency matrix this is. Rows are out-edges.\n"
          "Returns (ranks, info).")
      .def(
          "solve_cg",
          [](const Matrix& matrix, InputArray<Number> b, Number tolerance, int max_iterations) {
            if (b.size() != static_cast<size_t>(matrix.m)) {
              throw std::invalid_argument("right-hand side length must match the matrix");
            }
            spcraft::CgOptions<Number> options;
            options.tolerance = tolerance;
            options.max_iterations = max_iterations;

            Number* solution = new Number[matrix.n];
            std::fill(solution, solution + matrix.n, Number{0});
            const spcraft::DenseVector<Index, Number> rhs(const_cast<Number*>(b.data()),
                                                          static_cast<Index>(b.size()));
            spcraft::DenseVector<Index, Number> x(solution, matrix.n);
            const auto result = spcraft::cg_openmp(matrix, rhs, x, options);

            nb::dict info;
            info["iterations"] = result.iterations;
            info["residual"] = result.residual;
            info["converged"] = result.converged;
            info["breakdown"] = result.breakdown;
            return nb::make_tuple(AdoptAsNumpy(solution, static_cast<size_t>(matrix.n)), info);
          },
          "b"_a, "tolerance"_a = Number{1e-10}, "max_iterations"_a = 1000,
          "Solve A x = b by conjugate gradient. A must be symmetric positive\n"
          "definite with all nonzeros stored. Returns (x, info).")
      .def(
          "power_iteration",
          [](const Matrix& matrix, Number tolerance, int max_iterations) {
            spcraft::PowerIterationOptions<Number> options;
            options.tolerance = tolerance;
            options.max_iterations = max_iterations;

            Number* vector = new Number[matrix.n];
            std::fill(vector, vector + matrix.n, Number{0});  // seeded by the solver
            spcraft::DenseVector<Index, Number> x(vector, matrix.n);
            const auto result = spcraft::power_iteration_openmp(matrix, x, options);

            nb::dict info;
            info["iterations"] = result.iterations;
            info["residual"] = result.residual;
            info["converged"] = result.converged;
            return nb::make_tuple(result.eigenvalue,
                                  AdoptAsNumpy(vector, static_cast<size_t>(matrix.n)), info);
          },
          "tolerance"_a = Number{1e-10}, "max_iterations"_a = 1000,
          "Dominant eigenpair by power iteration.\n"
          "Returns (eigenvalue, eigenvector, info).")
      .def("__repr__", [name](const Matrix& matrix) {
        return std::string("spcraft.") + name + "(shape=(" + std::to_string(matrix.m) + ", " +
               std::to_string(matrix.n) + "), nnz=" + std::to_string(matrix.nnz) + ")";
      });
}

//! COO components are strided inside Entry; NumPy receives a contiguous owned copy.
template <std::size_t Component, class Number>
nb::object CooComponentToNumpy(const spcraft::CooMatrix<Index, Number, Offset>& matrix)
{
  static_assert(Component < 3, "COO entries have row, column and value components");
  using T = std::conditional_t<Component == 2, Number, Index>;
  const auto size = static_cast<std::size_t>(matrix.nnz);
  auto copy = std::make_unique<T[]>(size);
  for (std::size_t i = 0; i < size; ++i) {
    if constexpr (Component == 0) {
      copy[i] = matrix.entries[i].row;
    } else if constexpr (Component == 1) {
      copy[i] = matrix.entries[i].col;
    } else {
      copy[i] = matrix.entries[i].val;
    }
  }
  return AdoptAsNumpy(copy.release(), size);
}

template <class Number>
void BindCoo(nb::module_& module, const char* name)
{
  using Matrix = spcraft::CooMatrix<Index, Number, Offset>;

  nb::class_<Matrix>(module, name)
      .def_static("from_arrays", &CooFromArrays<Number>, "row_indices"_a, "column_indices"_a,
                  "values"_a, "rows"_a, "columns"_a,
                  "Create a matrix by copying one-dimensional NumPy arrays.")
      .def_static(
          "from_matrix_market", [](const std::string& filename) { return Matrix(filename); },
          "filename"_a)
      .def("clone", &Matrix::Clone)
      .def("to_csr", &Matrix::ToCsr)
      .def_prop_ro("shape", [](const Matrix& matrix) { return nb::make_tuple(matrix.m, matrix.n); })
      .def_ro("nnz", &Matrix::nnz)
      .def_prop_ro("row_indices",
                   [](const Matrix& matrix) { return CooComponentToNumpy<0>(matrix); })
      .def_prop_ro("column_indices",
                   [](const Matrix& matrix) { return CooComponentToNumpy<1>(matrix); })
      .def_prop_ro("values", [](const Matrix& matrix) { return CooComponentToNumpy<2>(matrix); })
      .def("__repr__", [name](const Matrix& matrix) {
        return std::string("spcraft.") + name + "(shape=(" + std::to_string(matrix.m) + ", " +
               std::to_string(matrix.n) + "), nnz=" + std::to_string(matrix.nnz) + ")";
      });
}

}  // namespace

NB_MODULE(_spcraft, module)
{
  module.doc() = "Nanobind interface to the SPCraft C++ library";

  BindCsr<float>(module, "CsrMatrixF32");
  BindCsr<double>(module, "CsrMatrixF64");
  BindCoo<float>(module, "CooMatrixF32");
  BindCoo<double>(module, "CooMatrixF64");

  module.def("_gen_er_graph_f32", &GenerateERGraph<float>, "vertices"_a, "expected_degree"_a,
             "seed"_a);
  module.def("_gen_er_graph_f64", &GenerateERGraph<double>, "vertices"_a, "expected_degree"_a,
             "seed"_a);
  module.def("_gen_rmat_f32", &GenerateRMAT<float>, "scale"_a, "edge_factor"_a, "seed"_a,
             "a"_a = 0.57, "b"_a = 0.19, "c"_a = 0.19, "d"_a = 0.05);
  module.def("_gen_rmat_f64", &GenerateRMAT<double>, "scale"_a, "edge_factor"_a, "seed"_a,
             "a"_a = 0.57, "b"_a = 0.19, "c"_a = 0.19, "d"_a = 0.05);
}
