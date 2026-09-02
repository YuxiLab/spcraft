#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <limits>
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
  if (nnz != 0) {
    std::copy(row_indices.data(), row_indices.data() + nnz, result.row_id);
    std::copy(column_indices.data(), column_indices.data() + nnz, result.col_id);
    std::copy(values.data(), values.data() + nnz, result.val);
  }
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
void BindCsr(nb::module_& module, const char* name)
{
  using Matrix = spcraft::CsrMatrix<Index, Number, Offset>;

  nb::class_<Matrix>(module, name)
      .def_static("from_arrays", &CsrFromArrays<Number>, "row_offsets"_a, "column_indices"_a,
                  "values"_a, "rows"_a, "columns"_a,
                  "Create a matrix by copying one-dimensional NumPy arrays.")
      .def("clone", &Matrix::Clone)
      .def_prop_ro("shape",
                   [](const Matrix& matrix) { return nb::make_tuple(matrix.m, matrix.n); })
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
      .def("spmv",
           [](const Matrix& matrix, InputArray<Number> x) {
             if (x.size() != static_cast<size_t>(matrix.n)) {
               throw std::invalid_argument("vector length must match matrix column dimension");
             }
             Number* y = new Number[matrix.m];
             spcraft::spmv_openmp(matrix, x.data(), y);
             nb::capsule owner(y, [](void* pointer) noexcept { delete[] static_cast<Number*>(pointer); });
             OutputArray<Number> array(y, {static_cast<size_t>(matrix.m)}, owner);
             return nb::cast(array, nb::rv_policy::reference);
           },
           "x"_a,
           "Multiply the sparse matrix by a dense vector x.")
      .def("__matmul__",
           [](const Matrix& matrix, InputArray<Number> x) {
             if (x.size() != static_cast<size_t>(matrix.n)) {
               throw std::invalid_argument("vector length must match matrix column dimension");
             }
             Number* y = new Number[matrix.m];
             spcraft::spmv_openmp(matrix, x.data(), y);
             nb::capsule owner(y, [](void* pointer) noexcept { delete[] static_cast<Number*>(pointer); });
             OutputArray<Number> array(y, {static_cast<size_t>(matrix.m)}, owner);
             return nb::cast(array, nb::rv_policy::reference);
           },
           "x"_a)
      .def("dot",
           [](const Matrix& matrix, InputArray<Number> x) {
             if (x.size() != static_cast<size_t>(matrix.n)) {
               throw std::invalid_argument("vector length must match matrix column dimension");
             }
             Number* y = new Number[matrix.m];
             spcraft::spmv_openmp(matrix, x.data(), y);
             nb::capsule owner(y, [](void* pointer) noexcept { delete[] static_cast<Number*>(pointer); });
             OutputArray<Number> array(y, {static_cast<size_t>(matrix.m)}, owner);
             return nb::cast(array, nb::rv_policy::reference);
           },
           "x"_a)
      .def("__repr__", [name](const Matrix& matrix) {
        return std::string("spcraft.") + name + "(shape=(" + std::to_string(matrix.m) + ", " +
               std::to_string(matrix.n) + "), nnz=" + std::to_string(matrix.nnz) + ")";
      });
}

template <class Number>
void BindCoo(nb::module_& module, const char* name)
{
  using Matrix = spcraft::CooMatrix<Index, Number, Offset>;

  nb::class_<Matrix>(module, name)
      .def_static("from_arrays", &CooFromArrays<Number>, "row_indices"_a, "column_indices"_a,
                  "values"_a, "rows"_a, "columns"_a,
                  "Create a matrix by copying one-dimensional NumPy arrays.")
      .def_static("from_matrix_market", &Matrix::FromMatrixMarket, "filename"_a)
      .def("clone", &Matrix::Clone)
      .def("to_csr", &Matrix::ToCsr)
      .def_prop_ro("shape",
                   [](const Matrix& matrix) { return nb::make_tuple(matrix.m, matrix.n); })
      .def_ro("nnz", &Matrix::nnz)
      .def_prop_ro("row_indices",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.row_id, static_cast<size_t>(matrix.nnz));
                   })
      .def_prop_ro("column_indices",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.col_id, static_cast<size_t>(matrix.nnz));
                   })
      .def_prop_ro("values",
                   [](const Matrix& matrix) {
                     return CopyToNumpy(matrix.val, static_cast<size_t>(matrix.nnz));
                   })
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
}
