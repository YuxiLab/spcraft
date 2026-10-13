#!/usr/bin/env bash
# Run from the project root, or pass the template and output paths explicitly.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/generate_clangd.sh [TEMPLATE [OUTPUT]]

Defaults: TEMPLATE=.clangd.in, OUTPUT=.clangd
Requires GNU envsubst (Ubuntu/Debian: sudo apt install gettext-base).
Automatic dependency discovery also uses GNU realpath (coreutils).

Example:
  CLANGD_COMPILATION_DATABASE=./build scripts/generate_clangd.sh

Optional overrides: export CUDAROOT, MKLROOT, CXXOPTSROOT, FMMROOT,
GOOGLEBENCHMARKROOT, GOOGLETESTROOT, or MPIROOT to use your own installations.
Unset roots are discovered from the selected build's CMakeCache.txt and _deps/.
Explicitly exported empty values are errors, and overrides take priority.
GOOGLETESTROOT may point to an install prefix or a source checkout.
Dependency paths are interpreted relative to each compile command's directory.
Any other variable referenced by the template must be exported and non-empty.
The template must contain exactly one CompilationDatabase entry, written as:
  CompilationDatabase: "${CLANGD_COMPILATION_DATABASE}"
Relative database paths are resolved from OUTPUT's parent directory.
The existing output is replaced only after all checks and substitution succeed.

OpenMP's omp.h is detected with the CMake compiler, CXX (one executable), or
c++/g++/clang++ on PATH, then linked into the selected build directory.
Set OPENMP_INCLUDE_DIR explicitly if compiler detection cannot find omp.h.
OpenMP_ROOT/include is also checked when OpenMP_ROOT is exported.
No linker flags are added: clangd only needs the header for parsing.
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

cache_value() {
    [[ -r "$database_base/CMakeCache.txt" ]] || return 0
    awk -v key="$1" 'index($0, key ":") == 1 { sub(/^[^=]*=/, ""); print; exit }' \
        "$database_base/CMakeCache.txt"
}

find_dependency_root() {
    local variable=$1 package='' config_key='' candidate parent value
    local candidates=() headers=()
    case "$variable" in
        CXXOPTSROOT) package=cxxopts; headers=(include/cxxopts.hpp) ;;
        FMMROOT) package=fast_matrix_market; headers=(include/fast_matrix_market/fast_matrix_market.hpp) ;;
        GOOGLEBENCHMARKROOT) package=benchmark; headers=(include/benchmark/benchmark.h) ;;
        GOOGLETESTROOT)
            package=googletest
            config_key=GTest_DIR
            headers=(include/gtest/gtest.h googletest/include/gtest/gtest.h)
            ;;
        CUDAROOT)
            value=$(cache_value CMAKE_CUDA_COMPILER)
            [[ -n "$value" ]] && candidates+=("$(dirname -- "$(dirname -- "$value")")")
            headers=(include/cuda_runtime.h)
            ;;
        MKLROOT)
            value=$(cache_value MKL_INCLUDE)
            [[ -n "$value" ]] && candidates+=("$(dirname -- "$value")")
            headers=(include/mkl.h)
            ;;
        MPIROOT)
            value=$(cache_value MPI_CXX_HEADER_DIR)
            [[ -n "$value" ]] && candidates+=("$(dirname -- "$value")")
            headers=(include/mpi.h)
            ;;
        *) return 1 ;;
    esac
    if [[ -n "$package" ]]; then
        # Prefer an installed package selected by CMake over leftover CPM sources.
        value=$(cache_value "${config_key:-${package}_DIR}")
        [[ -n "$value" ]] && candidates+=("$value")
        candidates+=("$database_base/_deps/${package}-src")
        value=$(cache_value "${package}_SOURCE_DIR")
        [[ -n "$value" ]] && candidates+=("$value")
        value=$(cache_value "CPM_PACKAGE_${package}_SOURCE_DIR")
        [[ -n "$value" ]] && candidates+=("$value")
    fi
    for candidate in "${candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        # Package config files may live under lib/, lib64/, or a multiarch directory.
        while :; do
            for value in "${headers[@]}"; do
                if [[ -f "$candidate/$value" && -r "$candidate/$value" ]]; then
                    command -v realpath >/dev/null || fail 'Install GNU coreutils for automatic dependency discovery.'
                    realpath --relative-to="$database_base" -- "$candidate"
                    return
                fi
            done
            parent=$(dirname -- "$candidate")
            [[ "$parent" != "$candidate" ]] || break
            candidate=$parent
        done
    done
    return 1
}

find_openmp_header() {
    local cache_compiler='' compiler header
    if [[ ${OPENMP_INCLUDE_DIR+x} ]]; then
        header="$OPENMP_INCLUDE_DIR/omp.h"
        [[ -n "$OPENMP_INCLUDE_DIR" && -f "$header" && -r "$header" ]] ||
            fail 'OPENMP_INCLUDE_DIR must contain a readable omp.h.'
        printf '%s\n' "$header"
        return
    fi
    if [[ -n ${OpenMP_ROOT:-} && -r "$OpenMP_ROOT/include/omp.h" ]]; then
        printf '%s\n' "$OpenMP_ROOT/include/omp.h"
        return
    fi
    cache_compiler=$(cache_value CMAKE_CXX_COMPILER)
    for compiler in "$cache_compiler" "${CXX:-}" c++ g++ clang++; do
        [[ -n "$compiler" ]] && command -v "$compiler" >/dev/null || continue
        header=$("$compiler" -print-file-name=include/omp.h 2>/dev/null) || continue
        if [[ -f "$header" && -r "$header" ]]; then
            printf '%s\n' "$header"
            return
        fi
        # Clang may find libomp via its normal include search rather than GCC's layout.
        header=$(printf '#include <omp.h>\n' |
            "$compiler" -fopenmp -E -x c++ - 2>/dev/null |
            sed -n 's/^# [0-9][0-9]* "\([^"]*\/omp\.h\)".*$/\1/p') || continue
        header=${header%%$'\n'*}
        if [[ -f "$header" && -r "$header" ]]; then
            printf '%s\n' "$header"
            return
        fi
    done
    return 1
}

if [[ ${1-} == --help || ${1-} == -h ]]; then
    usage
    exit 0
fi
[[ $# -le 2 ]] || { usage >&2; exit 1; }

template=${1:-.clangd.in}
output=${2:-.clangd}
command -v envsubst >/dev/null || fail 'envsubst is missing; install GNU gettext.'
[[ -f "$template" && -r "$template" ]] || fail "Cannot read template: $template"
[[ -d "$(dirname -- "$output")" ]] || fail "Output directory does not exist: $(dirname -- "$output")"
[[ ! -d "$output" ]] || fail "Output is a directory: $output"
[[ ! "$template" -ef "$output" ]] || fail 'Template and output must be different files.'

template_text=$(cat -- "$template")
variables=$(envsubst --variables "$template_text" | sort -u)

# Resolve the database first: it supplies the dependency locations for missing roots.
database_entries=$(grep -c '^[[:space:]]*CompilationDatabase:' "$template" || true)
[[ "$database_entries" == 1 ]] || fail 'Template must contain exactly one CompilationDatabase entry.'
grep -Eq '^[[:space:]]*CompilationDatabase: "\$\{CLANGD_COMPILATION_DATABASE\}"[[:space:]]*$' "$template" ||
    fail 'Use CompilationDatabase: "${CLANGD_COMPILATION_DATABASE}" in the template.'
database_dir=$(printenv CLANGD_COMPILATION_DATABASE) ||
    fail 'Export a non-empty value for CLANGD_COMPILATION_DATABASE.'
[[ -n "$database_dir" ]] || fail 'Export a non-empty value for CLANGD_COMPILATION_DATABASE.'
case "$database_dir" in
    *'"'*|*'\'*|*$'\n'*|*$'\r'*) fail 'Database directory contains characters requiring YAML escaping.' ;;
esac
case "$database_dir" in
    /*) database_base="$database_dir" ;;
    *) database_base="$(dirname -- "$output")/$database_dir" ;;
esac
database_file="$database_base/compile_commands.json"
[[ -f "$database_file" && -r "$database_file" && -s "$database_file" ]] ||
    fail "Missing, unreadable, or empty compilation database: $database_file"

missing=0
while IFS= read -r variable; do
    [[ -n "$variable" ]] || continue
    # envsubst sees exported variables only, so shell-local values do not count.
    if ! value=$(printenv "$variable"); then
        if value=$(find_dependency_root "$variable"); then
            export "$variable=$value"
            printf 'Detected %s=%s\n' "$variable" "$value"
        else
            printf 'Error: Cannot discover %s from the build; export a non-empty value.\n' "$variable" >&2
            missing=1
            continue
        fi
    fi
    if [[ -z "$value" ]]; then
        printf 'Error: Export a non-empty value for %s.\n' "$variable" >&2
        missing=1
        continue
    fi
    # Substituted paths sit in double-quoted YAML strings, without YAML escaping.
    case "$value" in
        *'"'*|*'\'*|*$'\n'*|*$'\r'*)
            printf 'Error: %s contains characters requiring YAML escaping.\n' "$variable" >&2
            missing=1
            ;;
    esac
done <<< "$variables"
[[ "$missing" == 0 ]] || exit 1

openmp_header=''
if [[ "$template_text" == *clangd-openmp/include* ]]; then
    openmp_header=$(find_openmp_header) ||
        fail 'Cannot locate omp.h; set OPENMP_INCLUDE_DIR to its include directory.'
    # Compiler queries can return relative paths; symlinks need a stable target.
    openmp_header="$(cd -- "$(dirname -- "$openmp_header")" && pwd -P)/omp.h"
fi

# A temporary file beside the output permits an atomic replacement after success.
temporary=$(mktemp "${output}.tmp.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
envsubst "$template_text" < "$template" > "$temporary"
if [[ -n "$openmp_header" ]]; then
    openmp_support="$(dirname -- "$database_file")/clangd-openmp/include"
    mkdir -p -- "$openmp_support"
    ln -sfn -- "$openmp_header" "$openmp_support/omp.h"
    printf 'Found OpenMP header: %s\n' "$openmp_header"
fi
mv -f -- "$temporary" "$output"
printf 'Generated %s using %s\n' "$output" "$database_file"
