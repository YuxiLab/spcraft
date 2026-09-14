#include <CombBLAS/CombBLAS.h>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <mpi.h>

#include <chrono>
#include <fstream>
#include <memory>

#include "SpCraft.h"
#include "SpGEMMBenchmarkCommon.h"

namespace
{
using namespace spcraft::benchmarks;
using Comb = combblas::SpDCCols<int, double>;
using Tuples = combblas::SpTuples<int, double>;
using Ring = combblas::PlusTimesSRing<double, double>;

void Run(const cxxopts::ParseResult& options)
{
  const auto input = LoadMatrix<int, double, int>(options);
  const Eigen::SparseMatrix<double> e = ToEigen(*input.matrix);
  const auto csc = FromEigenColumns<int>(e);
  if (csc.m != csc.n || csc.m > std::numeric_limits<int>::max() / 107 ||
      ProductWork(*input.matrix, *input.matrix) >
          static_cast<std::uintmax_t>(std::numeric_limits<int>::max() / 2))
    throw std::invalid_argument("input must be square and within upstream hash limits");
  Tuples entries(csc.nnz, csc.m, csc.n);
  for (int col = 0; col < csc.n; ++col)
    for (int p = csc.col_ptr[col]; p < csc.col_ptr[col + 1]; ++p) {
      entries.rowindex(p) = csc.row_id[p];
      entries.colindex(p) = col;
      entries.numvalue(p) = csc.val[p];
    }
  const Comb a(entries, false);
  std::ofstream out(options["output"].as<std::string>());
  if (!out) throw std::runtime_error("cannot open CSV output");
  out.precision(17);
  out << "dataset,threads,iteration,variant,native_seconds,conversion_seconds,free_seconds,"
         "total_seconds\n";
  for (int threads : ParseThreadCounts(options["threads"].as<std::string>())) {
    OMP_SET_NUM_THREADS(threads);
    {
      const std::unique_ptr<Tuples> t(
          combblas::LocalSpGEMMHash<Ring, double>(a, a, false, false, true));
      const Comb serial(*t, false),
          parallel(csc.m, csc.n, static_cast<int>(t->getnnz()), t->tuples, false);
      if (serial.getnnz() != parallel.getnnz())
        throw std::runtime_error("conversion nnz mismatch");
      if (serial.getnnz()) {
        const auto* s = serial.GetDCSC();
        const auto* p = parallel.GetDCSC();
        if (s->nzc != p->nzc) throw std::runtime_error("conversion column count mismatch");
        for (int i = 0; i <= s->nzc; ++i)
          if (s->cp[i] != p->cp[i]) throw std::runtime_error("conversion offset mismatch");
        for (int i = 0; i < s->nzc; ++i)
          if (s->jc[i] != p->jc[i]) throw std::runtime_error("conversion column mismatch");
        for (int i = 0; i < t->getnnz(); ++i)
          if (s->ir[i] != p->ir[i] || s->numx[i] != p->numx[i])
            throw std::runtime_error("conversion entry mismatch");
      }
    }
    auto sample = [&](bool parallel, int iteration) {
      const auto start = std::chrono::steady_clock::now();
      const std::unique_ptr<Tuples> t(
          combblas::LocalSpGEMMHash<Ring, double>(a, a, false, false, true));
      const auto multiplied = std::chrono::steady_clock::now();
      auto convert_end = multiplied;
      {
        const Comb result =
            parallel ? Comb(csc.m, csc.n, static_cast<int>(t->getnnz()), t->tuples, false)
                     : Comb(*t, false);
        convert_end = std::chrono::steady_clock::now();
        if (result.getnnz() != t->getnnz()) std::abort();
      }
      const auto freed = std::chrono::steady_clock::now();
      if (iteration >= 0) {
        out << input.name << ',' << threads << ',' << iteration << ','
            << (parallel ? "parallel" : "serial") << ','
            << std::chrono::duration<double>(multiplied - start).count() << ','
            << std::chrono::duration<double>(convert_end - multiplied).count() << ','
            << std::chrono::duration<double>(freed - convert_end).count() << ','
            << std::chrono::duration<double>(freed - start).count() << '\n';
      }
      // Tuple destruction is excluded equally; the earlier phase run measured it separately.
    };
    for (int i = 0; i < 2; ++i) {
      sample(false, -1);
      sample(true, -1);
    }
    for (int i = 0; i < options["iterations"].as<int>(); ++i) {
      sample(i % 2 != 0, i);
      sample(i % 2 == 0, i);
    }
    out.flush();
    fmt::print("{} {} threads: serial/parallel conversion matched every entry\n", input.name,
               threads);
    std::fflush(stdout);
  }
}
}  // namespace

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int status = 0;
  try {
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 1 || provided < MPI_THREAD_FUNNELED)
      throw std::runtime_error("requires one rank with FUNNELED");
    cxxopts::Options options("spgemm_conversion_benchmark",
                             "Compare upstream serial/parallel tuple compression");
    AddCommonOptions(options, "int32", "int32");
    const auto parsed = options.parse(argc, argv);
    if (parsed.count("help"))
      fmt::print("{}\n", options.help());
    else {
      if (parsed["output"].as<std::string>().empty() || parsed["iterations"].as<int>() < 1)
        throw std::invalid_argument("output and positive iterations required");
      OMP_SET_DYNAMIC(0);
      Run(parsed);
    }
  } catch (const std::exception& e) {
    fmt::print(stderr, "error: {}\n", e.what());
    status = 1;
  }
  MPI_Finalize();
  return status;
}
