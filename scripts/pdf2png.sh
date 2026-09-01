#!/usr/bin/env bash
# ==============================================================================
# Script: pdf2png.sh
# Description: Converts PDF file(s) or entire directory of PDFs to high-resolution
#              PNG images. Each output PNG is saved alongside its source PDF.
#
# Usage:
#   ./scripts/pdf2png.sh <folder_or_file> [folder_or_file2 ...]
#   ./scripts/pdf2png.sh latex/figures/obj
#   ./scripts/pdf2png.sh -R latex/figures
#   ./scripts/pdf2png.sh -r 300 latex/figures/obj/*.pdf
#
# Options:
#   -r, --dpi <int>    Resolution in DPI (default: 300)
#   -R, --recursive    Recursively search directories for PDFs
#   -h, --help         Show this help message
# ==============================================================================

set -euo pipefail

DPI=300
RECURSIVE=false
INPUT_TARGETS=()

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--dpi)
      if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
        DPI="$2"
        shift 2
      else
        echo "Error: --dpi requires a positive integer." >&2
        exit 1
      fi
      ;;
    -R|--recursive)
      RECURSIVE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $(basename "$0") [-r DPI] [-R] <folder_or_file> [...]"
      echo ""
      echo "Options:"
      echo "  -r, --dpi <int>   Image resolution in DPI (default: 300)"
      echo "  -R, --recursive   Recursively find PDFs in provided directories"
      echo "  -h, --help        Display this help message"
      echo ""
      echo "Examples:"
      echo "  $(basename "$0") latex/figures/obj              # Convert all PDFs in folder"
      echo "  $(basename "$0") -R latex/                      # Recursively convert all PDFs in latex/"
      echo "  $(basename "$0") latex/figures/obj/dcsc_fig2.pdf # Convert single file"
      echo "  $(basename "$0") -r 400 latex/figures/obj/*.pdf"
      exit 0
      ;;
    -*)
      echo "Error: Unknown option '$1'. Use --help for usage." >&2
      exit 1
      ;;
    *)
      INPUT_TARGETS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#INPUT_TARGETS[@]} -eq 0 ]]; then
  echo "Error: No files or folders provided." >&2
  echo "Usage: $(basename "$0") [-r DPI] [-R] <folder_or_file> [...]" >&2
  exit 1
fi

# Detect conversion tool
if command -v pdftoppm >/dev/null 2>&1; then
  TOOL="pdftoppm"
elif command -v sips >/dev/null 2>&1; then
  TOOL="sips"
elif command -v convert >/dev/null 2>&1; then
  TOOL="convert"
else
  echo "Error: Neither 'pdftoppm', 'sips', nor 'convert' (ImageMagick) was found in PATH." >&2
  exit 1
fi

# Collect all PDF files from targets
PDF_FILES=()

for target in "${INPUT_TARGETS[@]}"; do
  if [[ -d "$target" ]]; then
    # Target is a directory
    if [[ "$RECURSIVE" == "true" ]]; then
      while IFS= read -r -d '' file; do
        PDF_FILES+=("$file")
      done < <(find "$target" -type f -name "*.pdf" -print0)
    else
      while IFS= read -r -d '' file; do
        PDF_FILES+=("$file")
      done < <(find "$target" -maxdepth 1 -type f -name "*.pdf" -print0)
    fi
  elif [[ -f "$target" ]]; then
    if [[ "$target" =~ \.pdf$ ]] || [[ "$target" =~ \.PDF$ ]]; then
      PDF_FILES+=("$target")
    else
      echo "Warning: '$target' does not have a .pdf extension, skipping..." >&2
    fi
  else
    echo "Warning: Target not found '$target', skipping..." >&2
  fi
done

if [[ ${#PDF_FILES[@]} -eq 0 ]]; then
  echo "No PDF files found to convert."
  exit 0
fi

# Remove duplicate entries (compatible with macOS / Linux / older Bash)
UNIQUE_PDFS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && UNIQUE_PDFS+=("$line")
done < <(printf '%s\n' "${PDF_FILES[@]}" | sort -u)

echo "Found ${#UNIQUE_PDFS[@]} PDF file(s) to convert using $TOOL (${DPI} DPI)..."

SUCCESS_COUNT=0

for pdf_path in "${UNIQUE_PDFS[@]}"; do
  dir_name="$(dirname "$pdf_path")"
  base_name="$(basename "$pdf_path" | sed -e 's/\.[pP][dD][fF]$//')"
  prefix="${dir_name}/${base_name}"
  target_png="${prefix}.png"

  echo "  --> Converting: $pdf_path -> $target_png"

  case "$TOOL" in
    pdftoppm)
      pdftoppm -png -r "$DPI" -singlefile "$pdf_path" "$prefix"
      ;;
    sips)
      sips -s format png "$pdf_path" --out "$target_png" >/dev/null 2>&1
      ;;
    convert)
      convert -density "$DPI" "$pdf_path" -quality 100 "$target_png"
      ;;
  esac

  if [[ -f "$target_png" ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "  [Error] Failed to generate $target_png" >&2
  fi
done

echo "Done! Successfully converted $SUCCESS_COUNT / ${#UNIQUE_PDFS[@]} file(s)."
