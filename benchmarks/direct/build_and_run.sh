#!/usr/bin/env bash
set -euo pipefail

root=${1:?Pass the staged baseline root}
if [[ -z ${SLURM_JOB_ID:-} || $(hostname -s) != lair-g* ]]; then
  echo 'Builds and runs require a lair Slurm compute node.' >&2
  exit 1
fi
cd "$root"
mkdir -p deps install/superlu_mt install/pastix results
for package in packages/*.deb; do
  dpkg-deb -x "$package" deps
done

deps="$root/deps/usr"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib/x86_64-linux-gnu:$root/install/pastix/lib:$root/install/superlu_mt/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$deps/lib/x86_64-linux-gnu/pkgconfig:$root/install/pastix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
jobs=${SLURM_CPUS_PER_TASK:-16}

{
  hostname
  date -u
  lscpu
  gcc --version
  cmake --version
  cat /proc/self/status | grep Cpus_allowed_list
  sha256sum packages/*.deb
  cat harness/provenance.json
} > results/environment.txt

cmake -S src/superlu_mt -B build/superlu_mt \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$root/install/superlu_mt" \
  -DTHREAD_API=OPENMP -DBLA_VENDOR=OpenBLAS \
  -Denable_examples=OFF -Denable_tests=OFF -Denable_fortran=OFF
cmake --build build/superlu_mt --parallel "$jobs"
cmake --install build/superlu_mt

cmake -S src/pastix -B build/pastix \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$root/install/pastix" \
  -DCMAKE_PREFIX_PATH="$deps" \
  -DCMAKE_LIBRARY_PATH="$deps/lib/x86_64-linux-gnu" \
  -DCMAKE_INCLUDE_PATH="$deps/include" \
  -DLAPACKE_DIR="$deps" -DMETIS_DIR="$deps" \
  -DLAPACKE_INCDIR="$deps/include" -DLAPACKE_LIBDIR="$deps/lib/x86_64-linux-gnu" \
  -DMETIS_INCDIR="$deps/include" -DMETIS_LIBDIR="$deps/lib/x86_64-linux-gnu" \
  -DBLA_VENDOR=OpenBLAS -DGIT_SUBMODULE=OFF \
  -DPASTIX_INT64=OFF -DPASTIX_ORDERING_SCOTCH=OFF -DPASTIX_ORDERING_METIS=ON \
  -DPASTIX_BUILD_FORTRAN=OFF -DPASTIX_BUILD_PYTHON=OFF \
  -DPASTIX_BUILD_EXAMPLES=OFF -DPASTIX_BUILD_TESTS=OFF \
  -DPASTIX_WITH_MPI=OFF -DPASTIX_WITH_CUDA=OFF \
  -DPASTIX_WITH_STARPU=OFF -DPASTIX_WITH_PARSEC=OFF
cmake --build build/pastix --parallel "$jobs"
cmake --install build/pastix

cmake -S harness/benchmarks/direct -B build/baselines \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$root/install/pastix;$root/install/superlu_mt;$deps" \
  -DCMAKE_LIBRARY_PATH="$deps/lib/x86_64-linux-gnu" \
  -DCMAKE_INCLUDE_PATH="$deps/include" \
  -DCXXOPTS_INCLUDE_DIR="$root/harness/dependencies/include" \
  -DCMAKE_BUILD_RPATH="$root/install/pastix/lib;$root/install/superlu_mt/lib;$deps/lib/x86_64-linux-gnu"
cmake --build build/baselines --parallel "$jobs"
ldd build/baselines/superlu_lu_baseline > results/superlu_linkage.txt
ldd build/baselines/pastix_lu_baseline > results/pastix_linkage.txt
python3 harness/benchmarks/direct/run_baselines.py --bin-dir build/baselines --output results
