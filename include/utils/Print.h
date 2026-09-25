#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "utils/MatrixMacros.h"

namespace spcraft
{

SP_MAT_TEMP
class CsrMatrix;
SP_MAT_TEMP
class CscMatrix;
SP_MAT_TEMP
class CooMatrix;

/// Print up to count elements of a valid std::vector, DenseVector, or Eigen vector.
/// Storage must be host-accessible; device pointers are unsupported.
template <class Vector>
  requires requires(const Vector& vec, std::size_t i, std::ostream& out) {
    { vec.size() } -> std::convertible_to<std::size_t>;
    out << vec[i];
  }
void PrintVector(const Vector& vec, std::size_t count, std::string_view name,
                 std::ostream& out = std::cout)
{
  const std::size_t limit = std::min(count, static_cast<std::size_t>(vec.size()));
  out << name << ": [";
  for (std::size_t i = 0; i < limit; ++i) {
    if (i != 0) out << ", ";
    out << vec[i];
  }
  out << "]\n";
}

namespace detail
{

struct MatrixPrintBlock {
  std::int64_t row, col, rows, cols;
};

inline MatrixPrintBlock CheckedPrintBlock(std::int64_t m, std::int64_t n, std::int64_t row,
                                          std::int64_t col, std::int64_t rows, std::int64_t cols)
{
  if (m < 0 || n < 0 || row < 0 || col < 0 || rows < 0 || cols < 0)
    throw std::invalid_argument("PrintMatrix dimensions, starts, and counts must be nonnegative");
  if (row > m || col > n) throw std::out_of_range("PrintMatrix block starts outside the matrix");
  return {row, col, std::min(rows, m - row), std::min(cols, n - col)};
}

inline void PrintMatrixHeading(std::ostream& out, std::string_view name, std::int64_t m,
                               std::int64_t n, const MatrixPrintBlock& block)
{
  out << name << " (" << m << " x " << n << "), rows [" << block.row << ", "
      << block.row + block.rows << "), cols [" << block.col << ", " << block.col + block.cols
      << "):\n";
}

// Store only entries in the requested block, without changing the source matrix.
class SparseMatrixPreview
{
  struct Cell {
    std::string text;
    bool duplicate = false;
  };
  MatrixPrintBlock block;
  std::ostream& out;
  std::map<std::pair<std::int64_t, std::int64_t>, Cell> cells;

 public:
  SparseMatrixPreview(MatrixPrintBlock block_, std::ostream& out_) : block(block_), out(out_) {}

  template <class NT>
  void Add(std::int64_t row, std::int64_t col, const NT& value)
  {
    if (row < block.row || row >= block.row + block.rows || col < block.col ||
        col >= block.col + block.cols)
      return;
    std::ostringstream formatted;
    formatted.copyfmt(out);
    formatted << value;
    auto [it, inserted] = cells.try_emplace({row, col}, Cell{formatted.str()});
    if (!inserted) {
      auto& cell = it->second;
      // Preserve duplicate entries: combining them would assume an addition rule.
      if (cell.duplicate)
        cell.text.pop_back();
      else
        cell.text = "{" + cell.text;
      cell.text += ", " + formatted.str() + "}";
      cell.duplicate = true;
    }
  }

  void Write(std::string_view name, std::int64_t m, std::int64_t n) const
  {
    PrintMatrixHeading(out, name, m, n, block);
    if (block.rows == 0 || block.cols == 0) {
      out << "[]\n";
      return;
    }
    const auto row_width = std::to_string(block.row + block.rows - 1).size();
    auto width = std::to_string(block.col + block.cols - 1).size();
    for (const auto& [position, cell] : cells) width = std::max(width, cell.text.size());
    const auto print_cell = [&](std::string_view text) {
      out << ' ' << std::string(width - text.size(), ' ') << text;
    };
    out << std::string(row_width, ' ');
    for (std::int64_t col = block.col; col < block.col + block.cols; ++col)
      print_cell(std::to_string(col));
    out << '\n';
    for (std::int64_t row = block.row; row < block.row + block.rows; ++row) {
      const auto label = std::to_string(row);
      out << std::string(row_width - label.size(), ' ') << label;
      for (std::int64_t col = block.col; col < block.col + block.cols; ++col) {
        const auto it = cells.find({row, col});
        print_cell(it == cells.end() ? std::string_view(".") : std::string_view(it->second.text));
      }
      out << '\n';
    }
  }
};

}  // namespace detail

/// Print a zero-based sparse subblock; counts are clipped at the matrix edges.
/// Missing entries appear as '.', stored zeros as '0', and duplicates as '{a, b}'.
/// The matrix must be valid and its storage host-accessible.
template <class IT, class NT, class OT>
void PrintMatrix(const CsrMatrix<IT, NT, OT>& matrix, std::int64_t row, std::int64_t col,
                 std::int64_t rows, std::int64_t cols, std::string_view name,
                 std::ostream& out = std::cout)
{
  const auto block = detail::CheckedPrintBlock(matrix.m, matrix.n, row, col, rows, cols);
  detail::SparseMatrixPreview preview(block, out);
  if (matrix.nnz != 0 && block.cols != 0) {
    for (auto r = block.row; r < block.row + block.rows; ++r)
      for (OT p = matrix.row_ptr[r]; p < matrix.row_ptr[r + 1]; ++p)
        preview.Add(r, matrix.col_id[p], matrix.val[p]);
  }
  preview.Write(name, matrix.m, matrix.n);
}

/// CSC overload of the sparse subblock printer, with the same bounds and storage requirements.
template <class IT, class NT, class OT>
void PrintMatrix(const CscMatrix<IT, NT, OT>& matrix, std::int64_t row, std::int64_t col,
                 std::int64_t rows, std::int64_t cols, std::string_view name,
                 std::ostream& out = std::cout)
{
  const auto block = detail::CheckedPrintBlock(matrix.m, matrix.n, row, col, rows, cols);
  detail::SparseMatrixPreview preview(block, out);
  if (matrix.nnz != 0 && block.rows != 0) {
    for (auto c = block.col; c < block.col + block.cols; ++c)
      for (OT p = matrix.col_ptr[c]; p < matrix.col_ptr[c + 1]; ++p)
        preview.Add(matrix.row_id[p], c, matrix.val[p]);
  }
  preview.Write(name, matrix.m, matrix.n);
}

/// COO overload; entries need not be sorted and duplicate coordinates are preserved.
template <class IT, class NT, class OT>
void PrintMatrix(const CooMatrix<IT, NT, OT>& matrix, std::int64_t row, std::int64_t col,
                 std::int64_t rows, std::int64_t cols, std::string_view name,
                 std::ostream& out = std::cout)
{
  const auto block = detail::CheckedPrintBlock(matrix.m, matrix.n, row, col, rows, cols);
  detail::SparseMatrixPreview preview(block, out);
  if (block.rows != 0 && block.cols != 0) {
    for (OT p = 0; p < matrix.nnz; ++p) {
      const auto& entry = matrix.entries[p];
      preview.Add(entry.row, entry.col, entry.val);
    }
  }
  preview.Write(name, matrix.m, matrix.n);
}

/// Print an Eigen dense or sparse subblock using its native stream formatting.
/// No Eigen headers are needed here; the matrix supplies block() and operator<<.
template <class Matrix>
  requires requires(const Matrix& matrix, std::int64_t i, std::ostream& out) {
    { matrix.rows() } -> std::convertible_to<std::int64_t>;
    { matrix.cols() } -> std::convertible_to<std::int64_t>;
    out << matrix.block(i, i, i, i);
  }
void PrintMatrix(const Matrix& matrix, std::int64_t row, std::int64_t col, std::int64_t rows,
                 std::int64_t cols, std::string_view name, std::ostream& out = std::cout)
{
  const auto block = detail::CheckedPrintBlock(matrix.rows(), matrix.cols(), row, col, rows, cols);
  detail::PrintMatrixHeading(out, name, matrix.rows(), matrix.cols(), block);
  if (block.rows == 0 || block.cols == 0)
    out << "[]\n";
  else
    out << matrix.block(block.row, block.col, block.rows, block.cols) << '\n';
}

/// Print the top-left rows-by-cols block; use the longer overload to choose its origin.
template <class Matrix>
void PrintMatrix(const Matrix& matrix, std::int64_t rows, std::int64_t cols, std::string_view name,
                 std::ostream& out = std::cout)
{
  PrintMatrix(matrix, 0, 0, rows, cols, name, out);
}

}  // namespace spcraft
