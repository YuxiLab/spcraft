#!/usr/bin/env bash

#SBATCH --job-name=spcraft-build
#SBATCH --account=r01789
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=h100-debug
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

source_dir=.
build_dir=build
parallel_jobs=16
cmake_args=(
	-S "${source_dir}"
	-B "${build_dir}"
	-DSPCRAFT_USE_CUDA=ON
	-DSPCRAFT_USE_MPI=ON
	-DSPCRAFT_BUILD_EXAMPLES=ON
)

if [[ -n "${SPCRAFT_CUDA_ARCHITECTURES:-}" ]]; then
	cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${SPCRAFT_CUDA_ARCHITECTURES}")
fi

echo "Configuring SPCraft with CUDA and MPI in ${build_dir}"
cmake "${cmake_args[@]}"

echo "Building SPCraft with ${parallel_jobs} parallel jobs"
cmake --build "${build_dir}" --parallel "${parallel_jobs}"

echo "Build complete: ${build_dir}"
