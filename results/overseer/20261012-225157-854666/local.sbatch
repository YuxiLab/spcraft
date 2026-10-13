#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --job-name=SpMV_development-local
set -uo pipefail
cd /home/yuxihong/workspace/code/spcraft || exit 1
mkdir -p build/overseer-dev/.cmake/api/v1/query || exit 1
: > build/overseer-dev/.cmake/api/v1/query/codemodel-v2
printf '%s\n' '@@OVERSEER {"event": "build", "server": "local", "argv": ["cmake", "--preset", "openmp-examples", "-B", "build/overseer-dev", "-DSPCRAFT_BUILD_BENCHMARKS=ON"]}'
if cmake --preset openmp-examples -B build/overseer-dev -DSPCRAFT_BUILD_BENCHMARKS=ON; then :; else
printf '%s\n' '@@OVERSEER {"event": "build_failed", "server": "local"}'
exit 1
fi
printf '%s\n' '@@OVERSEER {"event": "build", "server": "local", "argv": ["cmake", "--build", "build/overseer-dev", "--target", "bench_ompspmv", "--parallel", "4"]}'
if cmake --build build/overseer-dev --target bench_ompspmv --parallel 4; then :; else
printf '%s\n' '@@OVERSEER {"event": "build_failed", "server": "local"}'
exit 1
fi
failures=0
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0001 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 1, "server": "local", "dataset": "gupta3", "threads": 1, "method": "openmp-float"}'
env OMP_NUM_THREADS=1 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 1 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0001/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0001/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 1, "server": "local", "dataset": "gupta3", "threads": 1, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0002 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 2, "server": "local", "dataset": "gupta3", "threads": 1, "method": "openmp-double"}'
env OMP_NUM_THREADS=1 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 1 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0002/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0002/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 2, "server": "local", "dataset": "gupta3", "threads": 1, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0003 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 3, "server": "local", "dataset": "gupta3", "threads": 2, "method": "openmp-float"}'
env OMP_NUM_THREADS=2 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 2 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0003/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0003/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 3, "server": "local", "dataset": "gupta3", "threads": 2, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0004 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 4, "server": "local", "dataset": "gupta3", "threads": 2, "method": "openmp-double"}'
env OMP_NUM_THREADS=2 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 2 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0004/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0004/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 4, "server": "local", "dataset": "gupta3", "threads": 2, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0005 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 5, "server": "local", "dataset": "gupta3", "threads": 4, "method": "openmp-float"}'
env OMP_NUM_THREADS=4 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 4 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0005/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0005/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 5, "server": "local", "dataset": "gupta3", "threads": 4, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0006 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 6, "server": "local", "dataset": "gupta3", "threads": 4, "method": "openmp-double"}'
env OMP_NUM_THREADS=4 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/gupta3/gupta3.mtx --threads 4 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0006/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0006/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 6, "server": "local", "dataset": "gupta3", "threads": 4, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0007 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 7, "server": "local", "dataset": "pdb1HYS", "threads": 1, "method": "openmp-float"}'
env OMP_NUM_THREADS=1 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 1 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0007/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0007/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 7, "server": "local", "dataset": "pdb1HYS", "threads": 1, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0008 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 8, "server": "local", "dataset": "pdb1HYS", "threads": 1, "method": "openmp-double"}'
env OMP_NUM_THREADS=1 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 1 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0008/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0008/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 8, "server": "local", "dataset": "pdb1HYS", "threads": 1, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0009 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 9, "server": "local", "dataset": "pdb1HYS", "threads": 2, "method": "openmp-float"}'
env OMP_NUM_THREADS=2 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 2 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0009/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0009/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 9, "server": "local", "dataset": "pdb1HYS", "threads": 2, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0010 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 10, "server": "local", "dataset": "pdb1HYS", "threads": 2, "method": "openmp-double"}'
env OMP_NUM_THREADS=2 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 2 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0010/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0010/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 10, "server": "local", "dataset": "pdb1HYS", "threads": 2, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0011 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 11, "server": "local", "dataset": "pdb1HYS", "threads": 4, "method": "openmp-float"}'
env OMP_NUM_THREADS=4 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 4 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0011/result.json --precision float 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0011/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 11, "server": "local", "dataset": "pdb1HYS", "threads": 4, "method": "openmp-float", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
mkdir -p /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0012 || exit 1
printf '%s\n' '@@OVERSEER {"event": "start", "total": 12, "id": 12, "server": "local", "dataset": "pdb1HYS", "threads": 4, "method": "openmp-double"}'
env OMP_NUM_THREADS=4 /home/yuxihong/workspace/code/spcraft/build/overseer-dev/bin/bench_ompspmv --matrix /home/yuxihong/workspace/code/spcraft/dataset/output/pdb1HYS/pdb1HYS.mtx --threads 4 --index int32 --offset int32 --iterations 5 --max-time 0.05 --output /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0012/result.json --precision double 2>&1 | tee /home/yuxihong/workspace/code/spcraft/results/overseer/20261012-225157-854666/0012/output.log
code=${PIPESTATUS[0]}
printf '%s%s%s\n' '@@OVERSEER {"event": "finish", "id": 12, "server": "local", "dataset": "pdb1HYS", "threads": 4, "method": "openmp-double", "exit_code": ' "$code" '}'
if [ "$code" -ne 0 ]; then failures=$((failures+1)); fi
[ "$failures" -eq 0 ]
