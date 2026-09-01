#!/usr/bin/env bash
# ==============================================================================
# Script: build_figures.sh
# Description: Compiles all LaTeX figures in latex/figures/*.tex into PDFs
#              in latex/figures/obj/ and renders them as high-resolution PNGs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LATEX_DIR="${ROOT_DIR}/latex/figures"
OBJ_DIR="${LATEX_DIR}/obj"

mkdir -p "${OBJ_DIR}"

echo "Compiling LaTeX figures in ${LATEX_DIR}..."
for tex_file in "${LATEX_DIR}"/*.tex; do
    if [[ -f "${tex_file}" ]]; then
        base_name="$(basename "${tex_file}")"
        echo "  --> pdflatex: ${base_name}"
        pdflatex -interaction=nonstopmode -output-directory="${OBJ_DIR}" "${tex_file}" > /dev/null
    fi
done

echo "Converting PDFs to PNGs..."
"${SCRIPT_DIR}/pdf2png.sh" "${OBJ_DIR}"

echo "Copying PDFs and PNGs to output/..."
OUTPUT_DIR="${LATEX_DIR}/output"
mkdir -p "${OUTPUT_DIR}"
cp "${OBJ_DIR}"/*.pdf "${OUTPUT_DIR}/" 2>/dev/null || true
cp "${OBJ_DIR}"/*.png "${OUTPUT_DIR}/" 2>/dev/null || true

echo "All figures successfully built in ${OBJ_DIR} and copied to ${OUTPUT_DIR}"

