#include "utils/MatrixInputOptions.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <cxxopts.hpp>

namespace spcraft
{

void AddMatrixInputOptions(cxxopts::Options& parser)
{
  const std::shared_ptr<cxxopts::Value> index = cxxopts::value<std::string>()->default_value("int64");
  const std::shared_ptr<cxxopts::Value> value = cxxopts::value<std::string>()->default_value("double");
  const std::shared_ptr<cxxopts::Value> offset = cxxopts::value<std::string>();
  cxxopts::OptionAdder options = parser.add_options("Matrix input");
  options("matrix", "Matrix Market input file", cxxopts::value<std::string>());
  options("it", "Index type: int32 or int64", index);
  options("nt", "Value type: float or double", value);
  options("ot", "Offset type: int32 or int64 (default: matches --it)", offset);
}

MatrixInputOptions ReadMatrixInputOptions(const cxxopts::ParseResult& result)
{
  if (result.count("matrix") == 0) throw std::invalid_argument("Missing --matrix input file");
  MatrixInputOptions input;
  input.matrix_path = result["matrix"].as<std::string>();
  const std::string index = result["it"].as<std::string>();
  const std::string value = result["nt"].as<std::string>();
  const std::string offset = result.count("ot") ? result["ot"].as<std::string>() : index;

  if (index == "int32")
    input.index_type = MatrixIntegerType::kInt32;
  else if (index == "int64")
    input.index_type = MatrixIntegerType::kInt64;
  else
    throw std::invalid_argument("--it must be int32 or int64");

  if (value == "float")
    input.value_type = MatrixValueType::kFloat;
  else if (value == "double")
    input.value_type = MatrixValueType::kDouble;
  else
    throw std::invalid_argument("--nt must be float or double");

  if (offset == "int32")
    input.offset_type = MatrixIntegerType::kInt32;
  else if (offset == "int64")
    input.offset_type = MatrixIntegerType::kInt64;
  else
    throw std::invalid_argument("--ot must be int32 or int64");

  if (input.index_type == MatrixIntegerType::kInt64 && input.offset_type == MatrixIntegerType::kInt32) {
    throw std::invalid_argument("--it int64 requires --ot int64");
  }
  if (!std::filesystem::is_regular_file(input.matrix_path)) {
    throw std::invalid_argument("Matrix input is not a regular file: " + input.matrix_path);
  }
  return input;
}

}  // namespace spcraft
