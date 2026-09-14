#!/usr/bin/env bash

#SBATCH --job-name=spcraft-spmv
#SBATCH --account=r01789
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100-debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/spcraft/slurm/%x-%j.out

# Submit the OpenMP SpMV benchmark to SLURM. All of the SPCRAFT_* variables
# understood by spmv.sh can be exported before sbatch, or passed with
# --export, for example:
#
#   sbatch --export=ALL,SPCRAFT_MATRIX=dataset/suitsparse/cant/cant.mtx \
#          benchmarks/spcraft/slurm/spmv.sbatch

set -euo pipefail

repo_dir="${SPCRAFT_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" || ! -x "${repo_dir}/benchmarks/spcraft/slurm/spmv.sh" ]]; then
	echo "error: submit this job from the SPCraft repository root" >&2
	exit 1
fi

cd "${repo_dir}"

# The allocation, not the node, decides how many cores this job may use.
# --exclusive above keeps other jobs off the socket so the sweep is comparable.
export SPCRAFT_THREADS="${SPCRAFT_THREADS:-$(
	count=1
	list=""
	while ((count < SLURM_CPUS_PER_TASK)); do
		list+="${count},"
		count=$((count * 2))
	done
	echo "${list}${SLURM_CPUS_PER_TASK}"
)}"

echo "job ${SLURM_JOB_ID} on $(hostname)"
echo "cpus-per-task : ${SLURM_CPUS_PER_TASK}"
echo "thread sweep  : ${SPCRAFT_THREADS}"
echo

./benchmarks/spcraft/slurm/spmv.sh
