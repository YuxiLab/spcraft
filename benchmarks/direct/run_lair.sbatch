#!/usr/bin/env bash
#SBATCH --job-name=spcraft-direct-baseline
#SBATCH --account=yuxilab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --constraint=cpujobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
root=${1:?Pass the staged baseline root}
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  bash "$root/harness/benchmarks/direct/build_and_run.sh" "$root"
