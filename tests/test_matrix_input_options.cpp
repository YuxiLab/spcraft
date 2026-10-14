#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cxxopts.hpp>

#include "SpCraft.h"

namespace spcraft::tests
{
namespace
{

class MatrixInputOptionsTest : public ::testing::Test
{
 protected:
  /*****************************************************************************
   *                                Data Members                               *
   ****************************************************************************/
  std::filesystem::path matrix_path;

  /*****************************************************************************
   *                              Member Functions                             *
   ****************************************************************************/
  void SetUp() override
  {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto filename = "spcraft-input-options-" + std::to_string(stamp) + ".mtx";
    matrix_path = std::filesystem::temp_directory_path() / filename;
    std::ofstream output(matrix_path);
    output << "%%MatrixMarket matrix coordinate real general\n2 3 2\n1 1 2.5\n2 3 -4\n";
    ASSERT_TRUE(output.good()) << "Could not write " << matrix_path;
  }

  void TearDown() override { std::filesystem::remove(matrix_path); }

  cxxopts::ParseResult Parse(std::vector<std::string> arguments)
  {
    arguments.insert(arguments.begin(), "test_matrix_input_options");
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    cxxopts::Options parser(argv[0], "Matrix input test");
    AddMatrixInputOptions(parser);
    parser.parse_positional({"matrix"});
    return parser.parse(static_cast<int>(argv.size()), argv.data());
  }
};

TEST_F(MatrixInputOptionsTest, DefaultsUseInt64DoubleAndMatchingOffsets)
{
  const auto input = ReadMatrixInputOptions(Parse({matrix_path.string()}));
  EXPECT_EQ(input.matrix_path, matrix_path.string());
  EXPECT_EQ(input.index_type, MatrixIntegerType::kInt64);
  EXPECT_EQ(input.value_type, MatrixValueType::kDouble);
  EXPECT_EQ(input.offset_type, MatrixIntegerType::kInt64);
}

TEST_F(MatrixInputOptionsTest, IndexSelectionChangesDefaultOffsetType)
{
  const auto input =
      ReadMatrixInputOptions(Parse({"--matrix", matrix_path.string(), "--it", "int32"}));
  EXPECT_EQ(input.index_type, MatrixIntegerType::kInt32);
  EXPECT_EQ(input.offset_type, MatrixIntegerType::kInt32);
}

TEST_F(MatrixInputOptionsTest, OptionsComposeWithMainSpecificFlags)
{
  cxxopts::Options parser("other_main", "Composable options");
  AddMatrixInputOptions(parser);
  parser.add_options()("threads", "Thread count", cxxopts::value<int>());
  const auto path = matrix_path.string();
  const char* argv[] = {"other_main", "--matrix", path.c_str(), "--threads", "3", "--nt", "float"};
  const auto arguments = parser.parse(7, argv);
  const auto input = ReadMatrixInputOptions(arguments);
  EXPECT_EQ(arguments["threads"].as<int>(), 3);
  EXPECT_EQ(input.value_type, MatrixValueType::kFloat);
  EXPECT_EQ(input.matrix_path, path);
}

TEST_F(MatrixInputOptionsTest, EverySupportedCombinationDispatchesOnceAndLoadsTypedStorage)
{
  const std::pair<std::string, std::string> storage_pairs[] = {
      {"int32", "int32"}, {"int32", "int64"}, {"int64", "int64"}};
  for (const auto& [index, offset] : storage_pairs) {
    for (const std::string value : {"float", "double"}) {
      SCOPED_TRACE(index + "/" + value + "/" + offset);
      const auto arguments =
          Parse({matrix_path.string(), "--it", index, "--nt", value, "--ot", offset});
      const auto input = ReadMatrixInputOptions(arguments);
      int calls = 0;
      const auto read = [&]<class IT, class NT, class OT>() {
        ++calls;
        const auto matrix = ReadMatrixInput<IT, NT, OT>(input.matrix_path);
        EXPECT_EQ(matrix.m, 2);
        EXPECT_EQ(matrix.n, 3);
        EXPECT_EQ(matrix.nnz, 2);
        if (matrix.nnz == 2) {
          EXPECT_EQ(matrix.entries[0].row, 0);
          EXPECT_EQ(matrix.entries[0].col, 0);
          EXPECT_EQ(matrix.entries[0].val, NT{2.5});
          EXPECT_EQ(matrix.entries[1].row, 1);
          EXPECT_EQ(matrix.entries[1].col, 2);
          EXPECT_EQ(matrix.entries[1].val, NT{-4});
        }
        return std::tuple{sizeof(IT), std::is_same_v<NT, float>, sizeof(OT)};
      };
      const auto [index_bytes, is_float, offset_bytes] = DispatchMatrixInputTypes(input, read);
      EXPECT_EQ(calls, 1);
      EXPECT_EQ(index_bytes, index == "int32" ? sizeof(std::int32_t) : sizeof(std::int64_t));
      EXPECT_EQ(is_float, value == "float");
      EXPECT_EQ(offset_bytes, offset == "int32" ? sizeof(std::int32_t) : sizeof(std::int64_t));
    }
  }
}

TEST_F(MatrixInputOptionsTest, RejectsUnknownIntegerAndValueTypes)
{
  const std::pair<std::string, std::string> invalid_types[] = {
      {"--it", "uint32"}, {"--it", "int16"},  {"--nt", "bool"},
      {"--nt", "int32"},  {"--ot", "uint64"}, {"--ot", "int16"}};
  for (const auto& [flag, value] : invalid_types) {
    SCOPED_TRACE(flag + " " + value);
    const auto arguments = Parse({matrix_path.string(), flag, value});
    EXPECT_THROW(ReadMatrixInputOptions(arguments), std::invalid_argument);
  }
}

TEST_F(MatrixInputOptionsTest, RejectsInt64IndicesWithInt32Offsets)
{
  const auto arguments = Parse({matrix_path.string(), "--it", "int64", "--ot", "int32"});
  EXPECT_THROW(ReadMatrixInputOptions(arguments), std::invalid_argument);
  MatrixInputOptions input;
  input.offset_type = MatrixIntegerType::kInt32;
  const auto run = []<class IT, class NT, class OT>() { return 0; };
  EXPECT_THROW(DispatchMatrixInputTypes(input, run), std::invalid_argument);
}

TEST_F(MatrixInputOptionsTest, RejectsMissingMatrixOption)
{
  const auto arguments = Parse({"--it", "int32"});
  EXPECT_THROW(ReadMatrixInputOptions(arguments), std::invalid_argument);
}

TEST_F(MatrixInputOptionsTest, RejectsMissingFilesAndDirectories)
{
  const auto missing = Parse({matrix_path.string() + ".missing"});
  EXPECT_THROW(ReadMatrixInputOptions(missing), std::invalid_argument);
  const auto directory = Parse({matrix_path.parent_path().string()});
  EXPECT_THROW(ReadMatrixInputOptions(directory), std::invalid_argument);
}

TEST_F(MatrixInputOptionsTest, ConvenienceParserReportsHelpAndInvalidOptions)
{
  const char* help[] = {"example", "--help"};
  const auto help_result = ParseMatrixInput(2, help, "Example test");
  EXPECT_EQ(help_result.exit_code, 0);
  EXPECT_FALSE(help_result.options);

  const char* missing[] = {"example"};
  const auto missing_result = ParseMatrixInput(1, missing, "Example test");
  EXPECT_EQ(missing_result.exit_code, 1);
  EXPECT_FALSE(missing_result.options);

  const auto path = matrix_path.string();
  const char* invalid[] = {"example", path.c_str(), "--it", "uint32"};
  const auto invalid_result = ParseMatrixInput(4, invalid, "Example test");
  EXPECT_EQ(invalid_result.exit_code, 1);
  EXPECT_FALSE(invalid_result.options);

  const char* valid[] = {"example", path.c_str(), "--it", "int32", "--nt", "float"};
  const auto valid_result = ParseMatrixInput(6, valid, "Example test");
  ASSERT_TRUE(valid_result.options);
  EXPECT_EQ(valid_result.exit_code, 0);
  EXPECT_EQ(valid_result.options->index_type, MatrixIntegerType::kInt32);
  EXPECT_EQ(valid_result.options->value_type, MatrixValueType::kFloat);
  EXPECT_EQ(valid_result.options->offset_type, MatrixIntegerType::kInt32);
}

}  // namespace
}  // namespace spcraft::tests
