#!/usr/bin/env bash

#SBATCH --job-name=spcraft-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

repo_dir="${SPCRAFT_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" || ! -f "${repo_dir}/slurm/config_build.sh" ]]; then
	echo "error: submit this job from the SPCraft repository root" >&2
	exit 1
fi

cd "${repo_dir}"

export parallel_jobs="${SLURM_CPUS_PER_TASK}"

if [[ -z "${SPCRAFT_CUDA_ARCHITECTURES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
	detected_cuda_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
		| sed -n '1{s/\.//g;p;}' || true)"
	if [[ -n "${detected_cuda_arch}" ]]; then
		export SPCRAFT_CUDA_ARCHITECTURES="${detected_cuda_arch}"
	fi
fi

echo "Job ${SLURM_JOB_ID}: building on $(hostname)"
echo "CUDA architecture: ${SPCRAFT_CUDA_ARCHITECTURES:-compiler default}"

./slurm/config_build.sh
