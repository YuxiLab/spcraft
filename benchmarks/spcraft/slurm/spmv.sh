#!/usr/bin/env bash

# Run the SPCraft OpenMP SpMV benchmark from a terminal. Every setting is an
# environment variable with a default, so this script is also what the sbatch
# wrapper invokes. Paths stay relative to the repository root.
#
#   ./benchmarks/spcraft/slurm/spmv.sh
#   SPCRAFT_MATRIX=dataset/suitsparse/cant/cant.mtx ./benchmarks/spcraft/slurm/spmv.sh

set -euo pipefail

repo_dir="${SPCRAFT_SOURCE_DIR:-.}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" ]]; then
	echo "error: run this from the SPCraft repository root, or set SPCRAFT_SOURCE_DIR" >&2
	exit 1
fi

cd "${repo_dir}"

build_dir="${SPCRAFT_BUILD_DIR:-build}"
benchmark="${build_dir}/benchmarks/spcraft/spmv_openmp_benchmark"

if [[ ! -x "${benchmark}" ]]; then
	echo "error: ${benchmark} not found" >&2
	echo "build it with: cmake -S . -B ${build_dir} -DSPCRAFT_BUILD_BENCHMARKS=ON && cmake --build ${build_dir}" >&2
	exit 1
fi

# Empty SPCRAFT_MATRIX means the benchmark generates an ER graph instead.
matrix="${SPCRAFT_MATRIX:-}"
precision="${SPCRAFT_PRECISION:-both}"
threads="${SPCRAFT_THREADS:-}"
iterations="${SPCRAFT_ITERATIONS:-1000}"
max_time="${SPCRAFT_MAX_TIME:-10}"
seed="${SPCRAFT_SEED:-42}"
vertices="${SPCRAFT_VERTICES:-50000}"
degree="${SPCRAFT_DEGREE:-16}"
output_dir="${SPCRAFT_OUTPUT_DIR:-benchmarks/results/spmv_spcraft}"

# Pin threads so a sweep measures scaling rather than scheduler placement.
# Without this the per-iteration spread on a busy node dwarfs the signal.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

arguments=(
	--precision "${precision}"
	--iterations "${iterations}"
	--max-time "${max_time}"
	--seed "${seed}"
)

if [[ -n "${matrix}" ]]; then
	if [[ ! -f "${matrix}" ]]; then
		echo "error: matrix file not found: ${matrix}" >&2
		exit 1
	fi
	arguments+=(--matrix "${matrix}")
	label="$(basename "${matrix}" .mtx)"
else
	arguments+=(--vertices "${vertices}" --degree "${degree}")
	label="ER_V${vertices}_D${degree}"
fi

if [[ -n "${threads}" ]]; then
	arguments+=(--threads "${threads}")
fi

mkdir -p "${output_dir}"
# SLURM_JOB_ID is unset in a terminal run, so fall back to a timestamp.
run_id="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
output_file="${output_dir}/spmv-${label}-${run_id}.json"
arguments+=(--output "${output_file}")

echo "host          : $(hostname)"
echo "binary        : ${benchmark}"
echo "OMP_PROC_BIND : ${OMP_PROC_BIND}"
echo "OMP_PLACES    : ${OMP_PLACES}"
echo "raw log       : ${output_file}"
echo

"${benchmark}" "${arguments[@]}"
