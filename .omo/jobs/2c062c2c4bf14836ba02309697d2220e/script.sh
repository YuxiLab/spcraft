#!/usr/bin/env bash
set -uo pipefail
cd /home/yuxihong/workspace/code/sparse_computation/spcraft || exit 1
printf '%s\n' '@@OVERSEER {"event": "stage", "stage": "resources", "status": "SUCCESS", "server": "local", "allocation": "local"}'
mkdir -p build/cuda-openmp/.cmake/api/v1/query || exit 1
: > build/cuda-openmp/.cmake/api/v1/query/codemodel-v2
printf '%s\n' '@@OVERSEER {"event": "build", "server": "local", "argv": ["cmake", "--preset", "cuda-openmp", "-B", "build/cuda-openmp", "-DSPCRAFT_BUILD_BENCHMARKS=ON"]}'
if bash -c 'printf '"'"'%s\n'"'"' '"'"'@@OVERSEER {"event": "stage", "status": "RUNNING", "stage": "configure", "server": "local", "allocation": "local"}'"'"'
cmake --preset cuda-openmp -B build/cuda-openmp -DSPCRAFT_BUILD_BENCHMARKS=ON'; then
printf '%s\n' '@@OVERSEER {"event": "stage", "status": "SUCCESS", "stage": "configure", "server": "local", "allocation": "local"}'
else
printf '%s\n' '@@OVERSEER {"event": "stage", "status": "FAILURE", "stage": "configure", "server": "local", "allocation": "local"}'
printf '%s\n' '@@OVERSEER {"event": "build_failed", "server": "local"}'
exit 1
fi
printf '%s\n' '@@OVERSEER {"event": "built", "server": "local"}'
