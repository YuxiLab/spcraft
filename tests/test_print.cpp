#include <cstdint>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include "SpCraft.h"

namespace
{

int failures = 0;
using Dense = spcraft::DenseVector<std::int32_t, double>;
static_assert(std::is_same_v<decltype(std::declval<Dense&>()[0]), double&>);
static_assert(std::is_same_v<decltype(std::declval<const Dense&>()[0]), const double&>);

template <class Vector>
void CheckPrint(const Vector& vec, std::size_t count, std::string_view name,
                std::string_view expected)
{
  std::ostringstream out;
  spcraft::PrintVector(vec, count, name, out);
  if (out.str() != expected) {
    std::cerr << "PrintVector " << name << ": expected \"" << expected << "\", got \"" << out.str()
              << "\"\n";
    ++failures;
  }
}

}  // namespace

int main()
{
  const std::vector<int> standard{1, 2, 3};
  CheckPrint(standard, 2, "std prefix", "std prefix: [1, 2]\n");
  CheckPrint(standard, 3, "std full", "std full: [1, 2, 3]\n");
  CheckPrint(standard, 5, "std short", "std short: [1, 2, 3]\n");
  CheckPrint(standard, 0, "zero", "zero: []\n");
  CheckPrint(std::vector<int>{}, 5, "std empty", "std empty: []\n");
  CheckPrint(std::vector<bool>{true, false}, 5, "bool", "bool: [1, 0]\n");
  CheckPrint(standard, 1, std::string_view("label_suffix", 5), "label: [1]\n");

  Dense owned(3);
  owned[0] = 1.25;
  owned[1] = 2.5;
  owned[2] = -3.75;
  CheckPrint(owned, 2, "dense prefix", "dense prefix: [1.25, 2.5]\n");
  CheckPrint(owned, 5, "dense full", "dense full: [1.25, 2.5, -3.75]\n");
  CheckPrint(owned, 0, "dense zero", "dense zero: []\n");
  CheckPrint(Dense{}, 5, "dense empty", "dense empty: []\n");

  double buffer[]{4, 5, 6};
  spcraft::DenseVector<unsigned, double> view(buffer, 3);
  view[1] = 7;
  CheckPrint(view, 5, "dense view", "dense view: [4, 7, 6]\n");
  if (buffer[1] != 7) {
    std::cerr << "DenseVector view: expected buffer[1] = 7, got " << buffer[1] << '\n';
    ++failures;
  }
  Dense moved(std::move(owned));
  CheckPrint(owned, 5, "moved from", "moved from: []\n");
  CheckPrint(moved, 5, "moved to", "moved to: [1.25, 2.5, -3.75]\n");

  Eigen::VectorXd eigen(3);
  eigen << 1.25, 2.5, -3.75;
  CheckPrint(eigen, 2, "eigen prefix", "eigen prefix: [1.25, 2.5]\n");
  CheckPrint(eigen, 5, "eigen full", "eigen full: [1.25, 2.5, -3.75]\n");
  CheckPrint(eigen, 0, "eigen zero", "eigen zero: []\n");
  CheckPrint(Eigen::VectorXd{}, 5, "eigen empty", "eigen empty: []\n");
  const Eigen::RowVector3i row(1, 2, 3);
  CheckPrint(row, 5, "eigen row", "eigen row: [1, 2, 3]\n");
  // A row in a column-major matrix is not contiguous in memory.
  Eigen::Matrix<double, 3, 2> matrix;
  matrix << 1, 4, 2, 5, 3, 6;
  CheckPrint(matrix.row(1), 5, "strided row", "strided row: [2, 5]\n");
  CheckPrint(eigen.head(2) + eigen.tail(2), 5, "expression", "expression: [3.75, -1.25]\n");
  CheckPrint(Eigen::Map<const Eigen::Vector3d>(buffer), 5, "map", "map: [4, 7, 6]\n");

  if (failures != 0) {
    std::cerr << failures << " printing check(s) failed\n";
    return 1;
  }
  std::cout << "PrintVector checks passed\n";
  return 0;
}
