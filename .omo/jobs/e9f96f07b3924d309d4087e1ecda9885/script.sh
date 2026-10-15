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
printf '%s\n' '@@OVERSEER {"event": "build", "server": "local", "argv": ["cmake", "--build", "build/cuda-openmp", "--target", "ex02_spgemm", "--parallel", "4"]}'
if bash -c 'printf '"'"'%s\n'"'"' '"'"'@@OVERSEER {"event": "stage", "status": "RUNNING", "stage": "build", "server": "local", "allocation": "local"}'"'"'
cmake --build build/cuda-openmp --target ex02_spgemm --parallel 4'; then
printf '%s\n' '@@OVERSEER {"event": "stage", "status": "SUCCESS", "stage": "build", "server": "local", "allocation": "local"}'
else
printf '%s\n' '@@OVERSEER {"event": "stage", "status": "FAILURE", "stage": "build", "server": "local", "allocation": "local"}'
printf '%s\n' '@@OVERSEER {"event": "build_failed", "server": "local"}'
exit 1
fi
printf '%s\n' '@@OVERSEER {"event": "stage", "stage": "run", "status": "PENDING", "server": "local", "allocation": "local"}'
failures=0
mkdir -p /home/yuxihong/workspace/code/sparse_computation/spcraft/results/overseer/20261015-120707-1.0679891667078e+14/0001 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 1, "id": 1, "server": "local", "parameters": {"dataset": "/home/yuxihong/workspace/code/sparse_computation/spcraft/datasets/output/pdb1HYS/pdb1HYS.mtx", "threads": 16, "method": "openmp-double"}, "choices": {"dataset": "pdb1HYS", "threads": "16", "method": "openmp-double"}}'
env OMP_NUM_THREADS=16 bash -c 'printf '"'"'%s\n'"'"' '"'"'@@OVERSEER {"event": "stage", "stage": "run", "status": "RUNNING", "server": "local", "allocation": "local"}'"'"'
printf '"'"'%s\n'"'"' '"'"'Run 1/1 on local
Directory: /home/yuxihong/workspace/code/sparse_computation/spcraft
Command: env OMP_NUM_THREADS=16 /home/yuxihong/workspace/code/sparse_computation/spcraft/build/cuda-openmp/bin/ex02_spgemm --matrix /home/yuxihong/workspace/code/sparse_computation/spcraft/datasets/output/pdb1HYS/pdb1HYS.mtx --it int32 --ot int32 --nt double'"'"'
exec /home/yuxihong/workspace/code/sparse_computation/spcraft/build/cuda-openmp/bin/ex02_spgemm --matrix /home/yuxihong/workspace/code/sparse_computation/spcraft/datasets/output/pdb1HYS/pdb1HYS.mtx --it int32 --ot int32 --nt double' 2>&1 | tee /home/yuxihong/workspace/code/sparse_computation/spcraft/results/overseer/20261015-120707-1.0679891667078e+14/0001/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 1, "server": "local", "parameters": {"dataset": "/home/yuxihong/workspace/code/sparse_computation/spcraft/datasets/output/pdb1HYS/pdb1HYS.mtx", "threads": 16, "method": "openmp-double"}, "choices": {"dataset": "pdb1HYS", "threads": "16", "method": "openmp-double"}, "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
if [ "$failures" -eq 0 ]; then
printf '%s\n' '@@OVERSEER {"event": "stage", "stage": "run", "status": "SUCCESS", "server": "local", "allocation": "local"}'
else
printf '%s\n' '@@OVERSEER {"event": "stage", "stage": "run", "status": "FAILURE", "server": "local", "allocation": "local"}'
exit 1
fi
