#!/usr/bin/env bash

# Run one library's SpMV benchmark over every matrix in a dataset directory and
# write the raw JSON reports under benchmarks/results/<library>/spmv/:
#
#   ./benchmarks/scripts/run_spmv.sh spcraft dataset/example
#   ./benchmarks/scripts/run_spmv.sh cusparse dataset/example
#
# Report names carry the machine and the thread sweep, so runs captured on a
# different host or at a different team size never overwrite one another and a
# figure can never silently mix two machines:
#
#   spmv_consph_i32-o64-fp64_wormhole_t1-2-4-8-16.json
#
# The tag after the dataset names the template parameters the drivers were
# instantiated with, so runs that differ only in index or offset width do not
# overwrite one another either.
#
# Override the defaults through the environment:
#   SPCRAFT_BUILD_DIR   build tree holding the benchmark binaries (default build)
#   SPCRAFT_RESULTS_DIR results root (default benchmarks/results)
#   SPCRAFT_MACHINE     machine tag (default: the short hostname)
#   SPCRAFT_THREADS     thread sweep for the CPU backends (default 1,2,4,8,16)
#   SPCRAFT_PRECISION   float | double | both (default double)
#   SPCRAFT_INDEX       int32 | int64 | both (default int32)
#   SPCRAFT_OFFSET      int32 | int64 | both (default: the widest the library takes)
#   SPCRAFT_MAX_TIME    wall-clock budget per configuration (default 2.0)
#   SPCRAFT_ITERATIONS  maximum timed iterations per configuration (default 500)

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "usage: $0 <spcraft|mkl|cusparse> [dataset-dir]" >&2
	exit 2
fi

library="$1"
dataset_dir="${2:-dataset/example}"

build_dir="${SPCRAFT_BUILD_DIR:-build}"
results_root="${SPCRAFT_RESULTS_DIR:-benchmarks/results}"
machine="${SPCRAFT_MACHINE:-$(hostname -s)}"
threads="${SPCRAFT_THREADS:-1,2,4,8,16}"
precision="${SPCRAFT_PRECISION:-double}"
index="${SPCRAFT_INDEX:-int32}"
max_time="${SPCRAFT_MAX_TIME:-2.0}"
iterations="${SPCRAFT_ITERATIONS:-500}"

# Offsets default per library rather than globally: oneMKL's CSR is MKL_INT on
# both arrays, and cuSPARSE requires one width for both, so only the SPCraft
# kernel accepts a 64-bit offset beside a 32-bit index.
case "${library}" in
	spcraft)
		binary="${build_dir}/benchmarks/spcraft/spmv_openmp_benchmark"
		offset="${SPCRAFT_OFFSET:-int64}"
		;;
	mkl)
		binary="${build_dir}/benchmarks/mkl/spmv_mkl_benchmark"
		offset="${SPCRAFT_OFFSET:-int32}"
		;;
	cusparse)
		binary="${build_dir}/benchmarks/cusparse/cusparse_spmv_benchmark"
		offset="${SPCRAFT_OFFSET:-int32}"
		;;
	*)
		echo "error: unknown library '${library}'; expected spcraft, mkl or cusparse" >&2
		exit 2
		;;
esac

if [[ ! -x "${binary}" ]]; then
	echo "error: benchmark binary not executable: ${binary}" >&2
	echo "       build it first, e.g. cmake --build ${build_dir} -j" >&2
	exit 1
fi
if [[ ! -d "${dataset_dir}" ]]; then
	echo "error: dataset directory not found: ${dataset_dir}" >&2
	exit 1
fi

# The parallel resource the run actually used. A GPU run is driven by one host
# thread, so its thread sweep says nothing; the device is the identity that
# matters, and it belongs in the machine tag.
if [[ "${library}" == "cusparse" ]]; then
	device="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
		tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"
	machine="${machine}-${device}"
	resource="t1"
	thread_arguments=()
else
	resource="t${threads//,/-}"
	thread_arguments=(--threads "${threads}")
fi

# The machine tag is a single filename field, so it must not carry the '_' that
# separates fields nor the '-' that separates thread counts within the resource tag.
machine="$(printf '%s' "${machine}" | tr '_' '-')"

case "${precision}" in
	float)  precision_tag="fp32" ;;
	double) precision_tag="fp64" ;;
	both)   precision_tag="fpall" ;;
	*)
		echo "error: precision must be 'float', 'double' or 'both'" >&2
		exit 2
		;;
esac

# "int32" -> "32", "both" -> "all": the same shorthand the report tables use.
width_tag() {
	case "$1" in
		int32) echo "32" ;;
		int64) echo "64" ;;
		both)  echo "all" ;;
		*)
			echo "error: index and offset must be 'int32', 'int64' or 'both'" >&2
			exit 2
			;;
	esac
}

type_tag="i$(width_tag "${index}")-o$(width_tag "${offset}")-${precision_tag}"

output_dir="${results_root}/${library}/spmv"
mkdir -p "${output_dir}"

mapfile -t matrices < <(find "${dataset_dir}" -name '*.mtx' | sort)
if [[ ${#matrices[@]} -eq 0 ]]; then
	echo "error: no .mtx files under ${dataset_dir}" >&2
	exit 1
fi

echo "library  : ${library}"
echo "binary   : ${binary}"
echo "machine  : ${machine}"
echo "types    : ${type_tag}"
echo "resource : ${resource}"
echo "datasets : ${#matrices[@]}"
echo "results  : ${output_dir}"
echo

failures=0
for matrix in "${matrices[@]}"; do
	dataset="$(basename "${matrix}" .mtx)"
	report="${output_dir}/spmv_${dataset}_${type_tag}_${machine}_${resource}.json"

	echo "=== ${dataset} ==="
	if ! "${binary}" --matrix "${matrix}" "${thread_arguments[@]}" \
		--precision "${precision}" --index "${index}" --offset "${offset}" \
		--max-time "${max_time}" --iterations "${iterations}" --output "${report}"; then
		echo "FAILED ${dataset}" >&2
		failures=$((failures + 1))
	fi
	echo
done

if ((failures > 0)); then
	echo "${failures} of ${#matrices[@]} datasets failed" >&2
	exit 1
fi
echo "all ${#matrices[@]} datasets completed"
