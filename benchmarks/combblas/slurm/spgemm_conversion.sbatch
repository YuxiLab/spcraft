#!/usr/bin/env bash
#SBATCH --job-name=spcraft-spgemm-conversion
#SBATCH --account=r01789
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/combblas/slurm/%x-%j.out

set -euo pipefail
cd "${SPCRAFT_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}"
module load cray-python/3.11.7
export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
module list
lscpu

cmake -S dependencies/combblas-src -B build_combblas \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DBUILD_TESTING=OFF
cmake --build build_combblas --target CombBLAS --parallel 8
mkdir -p build_spgemm
if [[ ! -e build_spgemm/dependencies ]]; then
  ln -s ../dependencies build_spgemm/dependencies
fi
cmake -S . -B build_spgemm \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_BUILD_TYPE=Release -DSPCRAFT_BUILD_EXAMPLES=OFF -DBUILD_TESTING=ON \
  -DSPCRAFT_BUILD_BENCHMARKS=ON -DSPCRAFT_USE_COMBBLAS=ON \
  -DCombBLAS_DIR=build_combblas/CombBLAS \
  -DSPCRAFT_COMBBLAS_REVISION=2381a1f7a028b9f56d8531c0b7b157ae2ab27572 \
  -DCPM_boost_preprocessor_SOURCE=dependencies/boost_preprocessor-src \
  -DCPM_Eigen3_SOURCE=dependencies/eigen3-src \
  -DCPM_fast_matrix_market_SOURCE=dependencies/fast_matrix_market-src \
  -DCPM_cxxopts_SOURCE=dependencies/cxxopts-src \
  -DCPM_fmt_SOURCE=dependencies/fmt-src
cmake --build build_spgemm --target spgemm_conversion_benchmark --parallel 8
output_dir="benchmarks/results/combblas/spgemm/conversion-bigred-${SLURM_JOB_ID}"
mkdir -p "${output_dir}"
sha256sum benchmarks/combblas/SpGEMMConversionBenchmark.cpp \
  build_spgemm/benchmarks/combblas/spgemm_conversion_benchmark > "${output_dir}/source-sha256.txt"
for matrix in ecology1 cant; do
  build_spgemm/benchmarks/combblas/spgemm_conversion_benchmark \
    --matrix "dataset/suitsparse/${matrix}/${matrix}.mtx" \
    --threads 16,128 --precision double --iterations 12 \
    --output "${output_dir}/${matrix}.csv"
done
