#include <pastix.h>
#include <spm.h>
#include <cxxopts.hpp>

#include "../direct/BenchmarkMatrix.h"

int main(int argc, char** argv)
{
  try {
    cxxopts::Options arguments("pastix_lu_baseline", "Double-precision sparse LU baseline");
    arguments.add_options()("matrix", "Matrix Market input", cxxopts::value<std::string>())(
        "threads", "Factorization threads", cxxopts::value<int>());
    const auto parsed = arguments.parse(argc, argv);
    if (!parsed.count("matrix") || !parsed.count("threads")) {
      throw std::runtime_error(arguments.help());
    }
    const int threads = parsed["threads"].as<int>();
    if (threads < 1) {
      throw std::runtime_error("Thread count must be positive");
    }
    const auto matrix = spcraft::benchmark::ReadMatrix(parsed["matrix"].as<std::string>());
    spmatrix_t spm;
    spmInit(&spm);
    spm.mtxtype = SpmGeneral;
    spm.flttype = SpmDouble;
    spm.fmttype = SpmCSC;
    spm.baseval = 0;
    spm.n = matrix.n;
    spm.nnz = static_cast<spm_int_t>(matrix.val.size());
    spm.dof = 1;
    spmUpdateComputedFields(&spm);
    spmAlloc(&spm);
    std::copy(matrix.col_ptr.begin(), matrix.col_ptr.end(), spm.colptr);
    std::copy(matrix.row_id.begin(), matrix.row_id.end(), spm.rowptr);
    std::copy(matrix.val.begin(), matrix.val.end(), static_cast<double*>(spm.values));
    pastix_int_t iparm[IPARM_SIZE];
    double dparm[DPARM_SIZE];
    pastixInitParam(iparm, dparm);
    iparm[IPARM_VERBOSE] = PastixVerboseNot;
    iparm[IPARM_THREAD_NBR] = threads;
    iparm[IPARM_SCHEDULER] = PastixSchedDynamic;
    iparm[IPARM_ORDERING] = PastixOrderMetis;
    iparm[IPARM_FACTORIZATION] = PastixFactLU;
    iparm[IPARM_COMPRESS_WHEN] = PastixCompressNever;
    pastix_data_t* solver = nullptr;
    std::vector<int> affinity(threads);
    for (int thread = 0; thread < threads; ++thread) {
      affinity[thread] = thread;
    }
    // Pin the main thread too: this standalone driver does not launch other runtimes.
    pastixInitWithAffinity(&solver, MPI_COMM_WORLD, iparm, dparm, affinity.data());
    auto start = spcraft::benchmark::Clock::now();
    int info = pastix_task_analyze(solver, &spm);
    const double analyze = spcraft::benchmark::Seconds(start);
    if (info != PASTIX_SUCCESS) {
      throw std::runtime_error("PaStiX analysis failed: " + std::to_string(info));
    }
    start = spcraft::benchmark::Clock::now();
    info = pastix_task_numfact(solver, &spm);
    const double factor = spcraft::benchmark::Seconds(start);
    if (info != PASTIX_SUCCESS) {
      throw std::runtime_error("PaStiX factorization failed: " + std::to_string(info));
    }
    auto x = matrix.rhs;
    start = spcraft::benchmark::Clock::now();
    info = pastix_task_solve(solver, matrix.n, 1, x.data(), matrix.n);
    const double solve = spcraft::benchmark::Seconds(start);
    if (info != PASTIX_SUCCESS) {
      throw std::runtime_error("PaStiX solve failed: " + std::to_string(info));
    }
    const bool passed = spcraft::benchmark::Report("pastix", "metis", matrix, x, threads, analyze, factor,
                                                   solve, iparm[IPARM_NNZEROS], iparm[IPARM_STATIC_PIVOTING]);
    pastixFinalize(&solver);
    spmExit(&spm);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
