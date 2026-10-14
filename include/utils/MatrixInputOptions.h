#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace cxxopts
{
class Options;
class ParseResult;
}  // namespace cxxopts

namespace spcraft
{

enum class MatrixIntegerType { kInt32, kInt64 };

enum class MatrixValueType { kFloat, kDouble };

struct MatrixInputOptions {
  /*****************************************************************************
   *                               Data Members                                *
   ****************************************************************************/
  std::string matrix_path;
  MatrixIntegerType index_type = MatrixIntegerType::kInt64;
  MatrixValueType value_type = MatrixValueType::kDouble;
  MatrixIntegerType offset_type = MatrixIntegerType::kInt64;
};

//! Add matrix and type flags to a parser that can also own other option groups.
void AddMatrixInputOptions(cxxopts::Options& parser);

//! Read and validate matrix options after the caller's single command-line parse.
MatrixInputOptions ReadMatrixInputOptions(const cxxopts::ParseResult& result);

//! Invoke a templated callable once as operator()<IT, NT, OT>().
//! Every specialization of the callable must return the same type.
template <class Function>
decltype(auto) DispatchMatrixInputTypes(const MatrixInputOptions& input, Function&& function)
{
  const auto dispatch_values = [&]<class IT, class OT>() -> decltype(auto) {
    if (input.value_type == MatrixValueType::kFloat) {
      return std::forward<Function>(function).template operator()<IT, float, OT>();
    }
    if (input.value_type == MatrixValueType::kDouble) {
      return std::forward<Function>(function).template operator()<IT, double, OT>();
    }
    throw std::invalid_argument("NT must be float or double");
  };

  // The compiled storage pairs exclude 64-bit indices with 32-bit offsets.
  if (input.index_type == MatrixIntegerType::kInt32) {
    if (input.offset_type == MatrixIntegerType::kInt32) {
      return dispatch_values.template operator()<std::int32_t, std::int32_t>();
    }
    if (input.offset_type == MatrixIntegerType::kInt64) {
      return dispatch_values.template operator()<std::int32_t, std::int64_t>();
    }
  }
  if (input.index_type == MatrixIntegerType::kInt64 &&
      input.offset_type == MatrixIntegerType::kInt64) {
    return dispatch_values.template operator()<std::int64_t, std::int64_t>();
  }
  throw std::invalid_argument("Unsupported IT/OT combination: int64 indices require int64 offsets");
}

}  // namespace spcraft
