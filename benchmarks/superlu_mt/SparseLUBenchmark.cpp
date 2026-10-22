#include <slu_mt_ddefs.h>
#include <cxxopts.hpp>

#include "../direct/BenchmarkMatrix.h"

int main(int argc, char** argv)
{
  try {
    cxxopts::Options arguments("superlu_lu_baseline", "Double-precision sparse LU baseline");
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
    auto matrix = spcraft::benchmark::ReadMatrix(parsed["matrix"].as<std::string>());
    static_assert(sizeof(int_t) == sizeof(int32_t));
    SuperMatrix a{};
    SuperMatrix ac{};
    SuperMatrix l{};
    SuperMatrix u{};
    SuperMatrix b{};
    auto x = matrix.rhs;
    std::vector<int_t> perm_c(matrix.n);
    std::vector<int_t> perm_r(matrix.n);
    dCreate_CompCol_Matrix(&a, matrix.n, matrix.n, static_cast<int_t>(matrix.val.size()), matrix.val.data(),
                           matrix.row_id.data(), matrix.col_ptr.data(), SLU_NC, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&b, matrix.n, 1, x.data(), matrix.n, SLU_DN, SLU_D, SLU_GE);
    Gstat_t stats{};
    superlumt_options_t options{};
    const int_t panel_size = sp_ienv(1);
    const int_t relax = sp_ienv(2);
    StatAlloc(matrix.n, threads, panel_size, relax, &stats);
    StatInit(matrix.n, threads, &stats);
    auto start = spcraft::benchmark::Clock::now();
    constexpr int mmd_ata_ordering = 1;
    get_perm_c(mmd_ata_ordering, &a, perm_c.data());
    pdgstrf_init(threads, DOFACT, NOTRANS, NO, panel_size, relax, 1.0, NO, 0.0, perm_c.data(), perm_r.data(),
                 nullptr, 0, &a, &ac, &options, &stats);
    const double analyze = spcraft::benchmark::Seconds(start);
    int_t info = 0;
    start = spcraft::benchmark::Clock::now();
    pdgstrf(&options, &ac, perm_r.data(), &l, &u, &stats, &info);
    const double factor = spcraft::benchmark::Seconds(start);
    if (info != 0) {
      throw std::runtime_error("SuperLU_MT factorization failed: " + std::to_string(info));
    }
    start = spcraft::benchmark::Clock::now();
    dgstrs(NOTRANS, &l, &u, perm_r.data(), perm_c.data(), &b, &stats, &info);
    const double solve = spcraft::benchmark::Seconds(start);
    if (info != 0) {
      throw std::runtime_error("SuperLU_MT solve failed: " + std::to_string(info));
    }
    const auto* l_store = static_cast<const SCPformat*>(l.Store);
    const auto* u_store = static_cast<const NCPformat*>(u.Store);
    const int64_t entries = static_cast<int64_t>(l_store->nnz) + u_store->nnz - matrix.n;
    const bool passed = spcraft::benchmark::Report("superlu_mt", "mmd_ata", matrix, x, threads, analyze,
                                                   factor, solve, entries, 0);
    pxgstrf_finalize(&options, &ac);
    Destroy_SuperNode_SCP(&l);
    Destroy_CompCol_NCP(&u);
    Destroy_SuperMatrix_Store(&a);  // CSC arrays and RHS are borrowed from vectors.
    Destroy_SuperMatrix_Store(&b);
    StatFree(&stats);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
