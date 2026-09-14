#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

#include "BenchmarkCommon.h"
#include "SpCraft.h"

namespace
{

using namespace spcraft::benchmarks;

template <class NT>
constexpr cudaDataType_t DataType();

template <>
constexpr cudaDataType_t DataType<float>()
{
  return CUDA_R_32F;
}

template <>
constexpr cudaDataType_t DataType<double>()
{
  return CUDA_R_64F;
}

//! The cusparseIndexType_t naming the width cuSPARSE should read an array at.
template <class T>
constexpr cusparseIndexType_t IndexType()
{
  static_assert(std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>,
                "cuSPARSE indexes CSR with 32- or 64-bit integers only");
  return std::is_same_v<T, std::int32_t> ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
}

void CheckCuda(cudaError_t status, std::string_view api)
{
  if (status != cudaSuccess) {
    throw std::runtime_error("CUDA error in " + std::string(api) + ": " +
                             cudaGetErrorString(status));
  }
}

void CheckCusparse(cusparseStatus_t status, std::string_view api)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE error in " + std::string(api) + ": " +
                             cusparseGetErrorString(status));
  }
}

template <class IT, class NT, class OT>
class CusparseSpmvState
{
 public:
  CusparseSpmvState(const spcraft::CsrMatrix<IT, NT, OT>& matrix, int seed)
  {
    static_assert(std::is_same_v<NT, float> || std::is_same_v<NT, double>,
                  "cuSPARSE SpMV benchmark supports float and double only");

    m_ = static_cast<std::size_t>(matrix.m);
    n_ = static_cast<std::size_t>(matrix.n);
    nnz_ = static_cast<std::size_t>(matrix.nnz);

    CheckCuda(cudaMalloc(&d_row_ptr_, (matrix.m + 1) * sizeof(OT)), "cudaMalloc(d_row_ptr)");
    CheckCuda(cudaMalloc(&d_col_id_, matrix.nnz * sizeof(IT)), "cudaMalloc(d_col_id)");
    CheckCuda(cudaMalloc(&d_val_, matrix.nnz * sizeof(NT)), "cudaMalloc(d_val)");

    CheckCuda(cudaMemcpy(d_row_ptr_, matrix.row_ptr, (matrix.m + 1) * sizeof(OT),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(row_ptr)");
    CheckCuda(
        cudaMemcpy(d_col_id_, matrix.col_id, matrix.nnz * sizeof(IT), cudaMemcpyHostToDevice),
        "cudaMemcpy(col_id)");
    CheckCuda(cudaMemcpy(d_val_, matrix.val, matrix.nnz * sizeof(NT), cudaMemcpyHostToDevice),
              "cudaMemcpy(val)");

    h_x_.Allocate(matrix.n);
    h_x_.Random(seed);
    if (h_x_.n > 0) {
      d_x_.assign(h_x_.val, h_x_.val + h_x_.n);
    }
    d_y_.resize(matrix.m, NT{0});

    CheckCusparse(cusparseCreate(&handle_), "cusparseCreate");
    CheckCusparse(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST),
                  "cusparseSetPointerMode");
    CheckCusparse(cusparseCreateCsr(&matA_, matrix.m, matrix.n, matrix.nnz, d_row_ptr_,
                                    d_col_id_, d_val_, IndexType<OT>(), IndexType<IT>(),
                                    CUSPARSE_INDEX_BASE_ZERO, DataType<NT>()),
                  "cusparseCreateCsr");
    CheckCusparse(cusparseCreateDnVec(&vec_x_, matrix.n, thrust::raw_pointer_cast(d_x_.data()),
                                      DataType<NT>()),
                  "cusparseCreateDnVec(vec_x)");
    CheckCusparse(cusparseCreateDnVec(&vec_y_, matrix.m, thrust::raw_pointer_cast(d_y_.data()),
                                      DataType<NT>()),
                  "cusparseCreateDnVec(vec_y)");

    CheckCusparse(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_,
                                          matA_, vec_x_, &beta_, vec_y_, DataType<NT>(),
                                          CUSPARSE_SPMV_ALG_DEFAULT, &d_buffer_size_),
                  "cusparseSpMV_bufferSize");
    if (d_buffer_size_ > 0) {
      CheckCuda(cudaMalloc(&d_buffer_, d_buffer_size_), "cudaMalloc(d_buffer)");
    }

    // Warmup run to ensure descriptors and buffers are ready before timing.
    CheckCusparse(
        cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matA_, vec_x_, &beta_,
                     vec_y_, DataType<NT>(), CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_),
        "cusparseSpMV warmup");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");
  }

  CusparseSpmvState(const CusparseSpmvState&) = delete;
  CusparseSpmvState& operator=(const CusparseSpmvState&) = delete;

  ~CusparseSpmvState()
  {
    if (d_buffer_) {
      cudaFree(d_buffer_);
    }
    if (vec_y_) {
      cusparseDestroyDnVec(vec_y_);
    }
    if (vec_x_) {
      cusparseDestroyDnVec(vec_x_);
    }
    if (matA_) {
      cusparseDestroySpMat(matA_);
    }
    if (handle_) {
      cusparseDestroy(handle_);
    }
    if (d_val_) {
      cudaFree(d_val_);
    }
    if (d_col_id_) {
      cudaFree(d_col_id_);
    }
    if (d_row_ptr_) {
      cudaFree(d_row_ptr_);
    }
  }

  void Run()
  {
    CheckCusparse(
        cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matA_, vec_x_, &beta_,
                     vec_y_, DataType<NT>(), CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_),
        "cusparseSpMV");
  }

  std::size_t bytes_per_run() const
  {
    return static_cast<std::size_t>(matrix_size_bytes()) +
           static_cast<std::size_t>(m_ + n_) * sizeof(NT);
  }

  //! Host copy of the input vector, for the Eigen cross-check.
  const spcraft::DenseVector<IT, NT>& x() const
  {
    return h_x_;
  }

  //! Copy the device result back to host storage.
  void CopyResultToHost(NT* destination) const
  {
    CheckCuda(cudaMemcpy(destination, thrust::raw_pointer_cast(d_y_.data()), m_ * sizeof(NT),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(y)");
  }

  std::size_t nnz() const
  {
    return nnz_;
  }
  std::size_t rows() const
  {
    return m_;
  }
  std::size_t cols() const
  {
    return n_;
  }

 private:
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnVecDescr_t vec_x_ = nullptr;
  cusparseDnVecDescr_t vec_y_ = nullptr;
  OT* d_row_ptr_ = nullptr;
  IT* d_col_id_ = nullptr;
  NT* d_val_ = nullptr;
  spcraft::DenseVector<IT, NT> h_x_;
  thrust::device_vector<NT> d_x_;
  thrust::device_vector<NT> d_y_;
  void* d_buffer_ = nullptr;
  std::size_t d_buffer_size_ = 0;
  std::size_t m_ = 0;
  std::size_t n_ = 0;
  std::size_t nnz_ = 0;
  NT alpha_ = static_cast<NT>(1.0);
  NT beta_ = static_cast<NT>(0.0);

  std::size_t matrix_size_bytes() const
  {
    return (m_ + 1) * sizeof(OT) + nnz_ * (sizeof(IT) + sizeof(NT)) +
           static_cast<std::size_t>(n_) * sizeof(NT) +
           static_cast<std::size_t>(m_) * sizeof(NT);
  }
};

/**
 * @brief True when cuSPARSE can describe a CSR matrix with these index types.
 *
 * cusparseCreateCsr takes the row-offset and column-index widths as separate
 * arguments, but rejects a matrix that mixes them: asking for 32-bit indices
 * with 64-bit offsets fails with CUSPARSE_STATUS_NOT_SUPPORTED. Four of the
 * eight shared variants are therefore skipped here rather than removed from the
 * sequence, so the constraint stays visible where it applies.
 */
template <class IT, class OT>
inline constexpr bool kCusparseRepresentable = std::is_same_v<IT, OT>;

//! Everything a variant needs that does not depend on the type combination.
struct Settings {
  int max_iterations = 0;
  double budget = 0.0;
  int seed = 0;
  bool header_printed = false;
};

//! Load, upload, verify and time one <IT, NT, OT> combination on the GPU.
template <class IT, class NT, class OT>
void RunVariant(const cxxopts::ParseResult& result, Settings& settings,
                spcraft::BenchmarkReport& report)
{
  if constexpr (kCusparseRepresentable<IT, OT>) {
    if (!VariantSelected<IT, NT, OT>(result)) return;

    auto input = LoadMatrix<IT, NT, OT>(result);
    const spcraft::CsrMatrix<IT, NT, OT>& A = *input.matrix;
    CusparseSpmvState<IT, NT, OT> state(A, settings.seed);

    spcraft::DenseVector<IT, NT> y(A.m);
    state.CopyResultToHost(y.val);
    const double error = VerifyAgainstEigen(A, state.x(), y.val, "cuSPARSE");

    if (!settings.header_printed) {
      report.AddInfo(input.name, fmt::format("rows {}, cols {}, nnz {}", A.m, A.n, A.nnz));
      report.PrintHeader();
      settings.header_printed = true;
    }

    // Each timed call includes the device synchronize, so the measurement covers
    // the kernel rather than just the asynchronous launch.
    const auto once = [&] {
      state.Run();
      CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    };
    const std::string label = fmt::format("SpMV/cuSPARSE/{}/{}/{}", input.name,
                                          IndexOffsetName<IT, OT>(), PrecisionName<NT>());
    const double per_iteration = WarmUp(once);
    const int iterations =
        PlanIterations(per_iteration, settings.max_iterations, settings.budget, label);

    spcraft::BenchmarkRun run;
    DescribeRun(run, A, input.name, "cuSPARSE");
    run.threads = 0;
    run.verification_error = error;
    run.seconds = spcraft::TimeIterations(once, iterations);
    report.Add(std::move(run));
  }
}

#define SPCRAFT_BENCHMARK_RUN_VARIANT(r, product) \
  RunVariant<BOOST_PP_SEQ_ENUM(product)>(result, settings, report);

}  // namespace

int main(int argc, char** argv)
{
  try {
    cxxopts::Options options("spmv_cusparse_benchmark",
                             "cuSPARSE SpMV benchmark, verified against Eigen");
    // cuSPARSE requires one width for both index arrays, so the shared 64-bit
    // offset default would pair with a 32-bit index and select nothing.
    AddCommonOptions(options, "int32", "int32");
    const auto result = options.parse(argc, argv);

    if (result.count("help") > 0) {
      fmt::print("{}\n", options.help());
      return 0;
    }

    Settings settings;
    settings.max_iterations = result["iterations"].as<int>();
    settings.budget = result["max-time"].as<double>();
    settings.seed = result["seed"].as<int>();

    spcraft::BenchmarkReport report("cuSPARSE SpMV");
    report.AddInfo("kernel", "SpMV (CSR, CUSPARSE_SPMV_ALG_DEFAULT)");
    report.AddInfo("reference", "Eigen sparse product, every row verified");
    report.AddInfo("index", result["index"].as<std::string>());
    report.AddInfo("numeric", result["precision"].as<std::string>());
    report.AddInfo("offset", result["offset"].as<std::string>());
    report.AddInfo("max iterations", std::to_string(settings.max_iterations));
    report.AddInfo("time budget", fmt::format("{:.2f}s", settings.budget));

    BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_BENCHMARK_RUN_VARIANT,
                                  SPCRAFT_BENCHMARK_TYPE_SEQUENCES)

    if (!settings.header_printed) {
      throw std::runtime_error(
          "no type combination selected; cuSPARSE requires --index and --offset to name "
          "the same width, so i32/o64 and i64/o32 do not exist");
    }
    report.PrintFooter();

    const auto output = result["output"].as<std::string>();
    if (!output.empty()) {
      report.WriteJson(output);
      fmt::print("raw per-iteration log written to {}\n", output);
    }
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
  return 0;
}

#undef SPCRAFT_BENCHMARK_RUN_VARIANT
