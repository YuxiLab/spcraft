#!/usr/bin/env bash
# ==============================================================================
# Script: render_notebooks.sh
# Description: Renders Jupyter Notebooks (.ipynb) or Quarto files (.qmd) to PDF
#              using Quarto and the project's local .venv environment.
#
# Usage:
#   ./scripts/render_notebooks.sh                        # Render all notebooks in notebook/
#   ./scripts/render_notebooks.sh path/to/notebook.ipynb # Render a specific notebook
#   ./scripts/render_notebooks.sh --no-execute           # Render without re-executing
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOK_DIR="${ROOT_DIR}/notebook"
VENV_DIR="${ROOT_DIR}/.venv"

# Locate Python in .venv or system
if [[ -f "${VENV_DIR}/bin/python" ]]; then
    export QUARTO_PYTHON="${VENV_DIR}/bin/python"
elif command -v python3 &>/dev/null; then
    export QUARTO_PYTHON="$(command -v python3)"
fi

# Locate Quarto binary
QUARTO_BIN=""
if command -v quarto &>/dev/null; then
    QUARTO_BIN="$(command -v quarto)"
elif [[ -x "/Users/yuxihong/Applications/quarto/bin/quarto" ]]; then
    QUARTO_BIN="/Users/yuxihong/Applications/quarto/bin/quarto"
elif [[ -x "/Applications/quarto/bin/quarto" ]]; then
    QUARTO_BIN="/Applications/quarto/bin/quarto"
else
    echo "❌ Error: Quarto CLI not found. Please install Quarto (https://quarto.org)." >&2
    exit 1
fi

# Default options
EXECUTE_FLAG="--execute"
FORMAT="pdf"
TARGET_FILES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-execute)
            EXECUTE_FLAG="--no-execute"
            shift
            ;;
        --execute)
            EXECUTE_FLAG="--execute"
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --to)
            FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [FILE...]"
            echo ""
            echo "Options:"
            echo "  --execute        Re-run all code cells during render (default)"
            echo "  --no-execute     Render using existing notebook cell outputs"
            echo "  --to, --format   Output format: pdf (default), html, beamer"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                      # Render all notebooks in notebook/"
            echo "  $0 notebook/ch1_sec1_coo_csc.ipynb      # Render single notebook"
            echo "  $0 --no-execute notebook/*.ipynb        # Fast render without re-running code"
            exit 0
            ;;
        *)
            TARGET_FILES+=("$1")
            shift
            ;;
    esac
done

# If no files specified, find all .ipynb files in notebook/
if [[ ${#TARGET_FILES[@]} -eq 0 ]]; then
    while IFS= read -r -d '' file; do
        # Ignore checkpoints
        if [[ "$file" != *".ipynb_checkpoints"* ]]; then
            TARGET_FILES+=("$file")
        fi
    done < <(find "${NOTEBOOK_DIR}" -maxdepth 1 -name "*.ipynb" -not -name ".*" -print0 | sort -z)
fi

if [[ ${#TARGET_FILES[@]} -eq 0 ]]; then
    echo "⚠️  No notebook files found to render."
    exit 0
fi

FORMAT_UPPER="$(echo "$FORMAT" | tr '[:lower:]' '[:upper:]')"
echo "========================================================="
echo "  🚀 Quarto Notebook Converter -> ${FORMAT_UPPER}"
echo "  🐍 Python: ${QUARTO_PYTHON}"
echo "  ⚙️  Execution: ${EXECUTE_FLAG}"
echo "========================================================="

SUCCESS_COUNT=0
TOTAL_COUNT=${#TARGET_FILES[@]}

for file in "${TARGET_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "⚠️  Skipping non-existent file: $file"
        continue
    fi

    if [[ ! -s "$file" ]]; then
        echo "⚠️  Skipping empty (0 bytes) file: $(basename "$file")"
        continue
    fi

    DIR_NAME="$(cd "$(dirname "$file")" && pwd)"
    BASE_NAME="$(basename "$file")"
    NAME_NO_EXT="${BASE_NAME%.*}"
    OUTPUT_DIR="${DIR_NAME}/output"

    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "📄 Rendering: ${BASE_NAME} ..."
    START_TIME=$(date +%s)

    # Run Quarto inside the file's parent directory to ensure relative paths & _quarto.yml work
    if (cd "${DIR_NAME}" && "${QUARTO_BIN}" render "${BASE_NAME}" ${EXECUTE_FLAG} --to "${FORMAT}" --output-dir "output"); then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        OUTPUT_PDF="${OUTPUT_DIR}/${NAME_NO_EXT}.${FORMAT}"
        if [[ -f "${OUTPUT_PDF}" ]]; then
            echo "   ✅ Generated: ${OUTPUT_PDF} (${ELAPSED}s)"
        else
            echo "   ✅ Finished in ${ELAPSED}s (output saved in ${OUTPUT_DIR})"
        fi
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "   ❌ Failed to render ${BASE_NAME}"
    fi
done

echo ""
echo "========================================================="
echo "  🎉 Completed: ${SUCCESS_COUNT}/${TOTAL_COUNT} rendered successfully."
echo "========================================================="
