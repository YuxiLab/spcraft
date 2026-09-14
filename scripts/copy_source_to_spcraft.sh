#!/usr/bin/env bash

# Copy this repository's source tree to the sibling ../spcraft directory.
# Generated files, local tooling state, LaTeX sources, and datasets are omitted.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="$(cd -- "${SOURCE_DIR}/.." && pwd)/spcraft"
DRY_RUN=false
FORCE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--force]

Copy the source tree from:
  ${SOURCE_DIR}
to:
  ${DEST_DIR}

If the destination is an existing Git checkout, its .git directory is kept and
the working tree is refreshed. The script refuses to overwrite local changes.
Use --force to replace destination changes and --dry-run to preview the copy.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v rsync >/dev/null 2>&1; then
    echo "Error: rsync is required." >&2
    exit 1
fi

if [[ -e "${DEST_DIR}" ]] && [[ ! -d "${DEST_DIR}" ]]; then
    echo "Error: destination exists but is not a directory: ${DEST_DIR}" >&2
    exit 1
fi

if [[ -d "${DEST_DIR}" ]] && [[ -n "$(find "${DEST_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ ! -d "${DEST_DIR}/.git" ]]; then
        echo "Error: refusing to overwrite a non-empty, non-Git directory: ${DEST_DIR}" >&2
        exit 1
    fi

    DEST_GIT_ROOT="$(git -C "${DEST_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ "${DEST_GIT_ROOT}" != "${DEST_DIR}" ]]; then
        echo "Error: destination is not the root of its Git checkout: ${DEST_DIR}" >&2
        exit 1
    fi

    if [[ "${FORCE}" != true ]] && [[ -n "$(git -C "${DEST_DIR}" status --porcelain --untracked-files=all)" ]]; then
        echo "Error: destination has local changes; refusing to overwrite them:" >&2
        git -C "${DEST_DIR}" status --short >&2
        echo "Re-run with --force only if those changes may be replaced." >&2
        exit 1
    fi
fi

RSYNC_FILTERS=(
    # Repository and editor metadata.
    --exclude=/.git/
    --exclude=/.github/
    --exclude=/.vscode/
    --exclude=/.idea/
    --exclude=/.clangd
    --exclude=/.latexindent.yaml
    --exclude=/.python-version
    --exclude='*.swp'
    --exclude='*~'
    --exclude=.DS_Store

    # Content intentionally left out of the source-only copy.
    --exclude=/latex/
    --exclude=/dataset/
    --exclude=/data/
    --exclude=/notebook/
    --exclude=/benchmarks/results/
    --exclude=/doxygen/html/
    --exclude=/doxygen/latex/
    --exclude=/doxygen/SPCraft-reference.pdf
    --exclude=/scripts/copy_source_to_spcraft.sh

    # Build trees, generated output, environments, and caches.
    --exclude=/build/
    --exclude='/build_*/'
    --exclude='/cmake-build-*/'
    --exclude=/bin/
    --exclude=/lib/
    --exclude=/dist/
    --exclude=/_skbuild/
    --exclude=/.venv/
    --exclude=/venv/
    --exclude=/.pytest_cache/
    --exclude=/.cache/
    --exclude='**/__pycache__/'
    --exclude='**/CMakeFiles/'
    --exclude='**/.quarto/'
    --exclude='**/obj/'
    --exclude='**/output/'
    --exclude='*.egg-info/'
    --exclude=compile_commands.json
    --exclude=CMakeCache.txt

    # Compiled/generated files and dataset formats, wherever they occur.
    --exclude='*.o'
    --exclude='*.obj'
    --exclude='*.a'
    --exclude='*.so'
    --exclude='*.dylib'
    --exclude='*.dll'
    --exclude='*.exe'
    --exclude='*.py[co]'
    --exclude='*.mtx'
    --exclude='*.csv'
    --exclude='*.parquet'
    --exclude='*.tar'
    --exclude='*.tar.gz'
    --exclude='*.zip'
    --exclude='*.pdf'
    --exclude='*.png'
    --exclude='*.jpg'
    --exclude='*.jpeg'
    --exclude='*.gif'
    --exclude='*.avif'
)

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/spcraft-source.XXXXXX")"
trap 'rm -rf -- "${STAGE_ROOT}"' EXIT
STAGE_DIR="${STAGE_ROOT}/source"
mkdir "${STAGE_DIR}"

# Build a filtered staging tree first. Syncing that tree to the destination lets
# --delete remove stale generated files without ever putting .git at risk.
rsync --archive "${RSYNC_FILTERS[@]}" "${SOURCE_DIR}/" "${STAGE_DIR}/"

RSYNC_ARGS=(
    --archive
    --itemize-changes
    --delete
    --exclude=/.git/
)

if [[ "${DRY_RUN}" == true ]]; then
    RSYNC_ARGS+=(--dry-run)
fi

rsync "${RSYNC_ARGS[@]}" "${STAGE_DIR}/" "${DEST_DIR}/"

if [[ "${DRY_RUN}" == true ]]; then
    echo "Dry run complete; nothing was copied."
else
    echo "Source copy created at ${DEST_DIR}"
fi
