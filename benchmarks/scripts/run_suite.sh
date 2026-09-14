#!/usr/bin/env bash

# Run a benchmark over every dataset in a manifest produced by
# suitsparse-downloader:
#
#   ssdl suite manifest suites.yaml williams2007 \
#       --binary build/benchmarks/spcraft/spmv_openmp_benchmark \
#       --download -o manifest.json -- --precision double
#   ./benchmarks/scripts/run_suite.sh manifest.json
#
# The manifest names the binary and its arguments, so this script stays generic
# across the SpMV, SpGEMM, cuSPARSE and MKL drivers.

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "usage: $0 <manifest.json> [output-dir]" >&2
	exit 2
fi

manifest="$1"
output_dir="${2:-${SPCRAFT_OUTPUT_DIR:-benchmarks/results/suite}}"

if [[ ! -f "${manifest}" ]]; then
	echo "error: manifest not found: ${manifest}" >&2
	exit 1
fi

# python3 rather than jq: it is already required by the manifest writer, so this
# adds no dependency. Records are NUL-delimited to survive paths with spaces.
read_records() {
	python3 -c '
import json, sys
document = json.load(open(sys.argv[1]))
what = sys.argv[2]
if what == "scalar":
    sys.stdout.write(str(document.get(sys.argv[3]) or ""))
elif what == "arguments":
    for argument in document.get("arguments") or []:
        sys.stdout.write(argument + "\0")
elif what == "datasets":
    for dataset in document.get("datasets") or []:
        sys.stdout.write(dataset["name"] + "\n" + dataset["path"] + "\0")
' "$@"
}

binary="$(read_records "${manifest}" scalar binary)"
suite="$(read_records "${manifest}" scalar suite)"
suite="${suite:-unnamed}"

if [[ -z "${binary}" ]]; then
	echo "error: manifest does not name a binary; pass --binary when writing it" >&2
	exit 1
fi
if [[ ! -x "${binary}" ]]; then
	echo "error: benchmark binary not executable: ${binary}" >&2
	exit 1
fi

extra_arguments=()
while IFS= read -r -d '' argument; do
	extra_arguments+=("${argument}")
done < <(read_records "${manifest}" arguments)

names=()
paths=()
while IFS= read -r -d '' record; do
	names+=("${record%%$'\n'*}")
	paths+=("${record#*$'\n'}")
done < <(read_records "${manifest}" datasets)

if [[ ${#paths[@]} -eq 0 ]]; then
	echo "error: manifest lists no datasets" >&2
	exit 1
fi

mkdir -p "${output_dir}"
run_id="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"

echo "suite    : ${suite}"
echo "binary   : ${binary}"
echo "datasets : ${#paths[@]}"
echo "results  : ${output_dir}"
echo

failures=0
for position in "${!paths[@]}"; do
	name="${names[${position}]}"
	path="${paths[${position}]}"

	if [[ ! -f "${path}" ]]; then
		echo "SKIP ${name}: ${path} is gone" >&2
		failures=$((failures + 1))
		continue
	fi

	log="${output_dir}/${suite}-${name}-${run_id}.json"
	echo "=== ${name} ==="
	if ! "${binary}" --matrix "${path}" "${extra_arguments[@]}" --output "${log}"; then
		echo "FAILED ${name}" >&2
		failures=$((failures + 1))
	fi
	echo
done

if ((failures > 0)); then
	echo "${failures} of ${#paths[@]} datasets failed" >&2
	exit 1
fi
echo "all ${#paths[@]} datasets completed"
