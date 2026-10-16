#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="output"
# The collection website redirects to a data server that currently returns 403.
SUITESPARSE_MM_URL="${SUITESPARSE_MM_URL:-https://www.cise.ufl.edu/research/sparse/MM}"
mkdir -p "${OUTPUT_DIR}"

# Helper to download a file using curl or wget
download_file() {
    local url="$1"
    local dest_file="$2"
    local partial_file="${dest_file}.part"
    mkdir -p "$(dirname "${dest_file}")"
    if command -v curl >/dev/null 2>&1; then
        if ! curl -fSL --retry 2 --connect-timeout 15 -o "${partial_file}" "${url}"; then
            rm -f "${partial_file}"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget --tries=3 --timeout=30 -O "${partial_file}" "${url}"; then
            rm -f "${partial_file}"
            return 1
        fi
    else
        echo "Error: neither curl nor wget found."
        return 1
    fi
    mv "${partial_file}" "${dest_file}"
}

# Function to download a dataset
download_dataset() {
    local dataset_name="$1"
    local url="$2"

    if [ -s "${OUTPUT_DIR}/${dataset_name}/${dataset_name}.mtx" ]; then
        echo "${OUTPUT_DIR}/${dataset_name} dataset exists."
    else
        echo "Downloading ${dataset_name} to ${OUTPUT_DIR}..."
        rm -f "${OUTPUT_DIR}/${dataset_name}.tar.gz"
        download_file "${url}" "${OUTPUT_DIR}/${dataset_name}.tar.gz" || return 1
        tar -xzf "${OUTPUT_DIR}/${dataset_name}.tar.gz" -C "${OUTPUT_DIR}" || return 1
        rm -f "${OUTPUT_DIR}/${dataset_name}.tar.gz"
    fi
}

download_tiny() {
    echo "Downloading tiny dataset to ${OUTPUT_DIR}"
    # tilespgemm
    download_dataset pdb1HYS "${SUITESPARSE_MM_URL}/Williams/pdb1HYS.tar.gz"
    download_dataset consph "${SUITESPARSE_MM_URL}/Williams/consph.tar.gz"
    download_dataset webbase-1M "${SUITESPARSE_MM_URL}/Williams/webbase-1M.tar.gz"
    download_dataset bcspwr06 "${SUITESPARSE_MM_URL}/HB/bcspwr06.tar.gz"
}

download_small() {
    echo "Downloading small dataset to ${OUTPUT_DIR}"
    # tilespgemm
    download_dataset pdb1HYS "${SUITESPARSE_MM_URL}/Williams/pdb1HYS.tar.gz"
    download_dataset consph "${SUITESPARSE_MM_URL}/Williams/consph.tar.gz"
    download_dataset webbase-1M "${SUITESPARSE_MM_URL}/Williams/webbase-1M.tar.gz"
    download_dataset af_shell10 "${SUITESPARSE_MM_URL}/Schenk_AFE/af_shell10.tar.gz"
    download_dataset SiO2 "${SUITESPARSE_MM_URL}/PARSEC/SiO2.tar.gz"
    download_dataset gupta3 "${SUITESPARSE_MM_URL}/Gupta/gupta3.tar.gz"
    download_dataset rma10 "${SUITESPARSE_MM_URL}/Bova/rma10.tar.gz"
    download_dataset TSOPF_FS_b300_c2 "${SUITESPARSE_MM_URL}/TSOPF/TSOPF_FS_b300_c2.tar.gz"
}

download_medium() {
    echo "Downloading medium dataset to ${OUTPUT_DIR}"
    # SpGEMM/SpMV medium benchmark matrices (~50MB-300MB uncompressed / ~10MB-120MB compressed)
    download_dataset cop20k_A "${SUITESPARSE_MM_URL}/Williams/cop20k_A.tar.gz"
    download_dataset cant "${SUITESPARSE_MM_URL}/Williams/cant.tar.gz"
    download_dataset mac_econ_fwd500 "${SUITESPARSE_MM_URL}/Williams/mac_econ_fwd500.tar.gz"
    download_dataset pwtk "${SUITESPARSE_MM_URL}/Boeing/pwtk.tar.gz"
    download_dataset hood "${SUITESPARSE_MM_URL}/GHS_psdef/hood.tar.gz"
    download_dataset bmwcra_1 "${SUITESPARSE_MM_URL}/GHS_psdef/bmwcra_1.tar.gz"
    download_dataset web-Google "${SUITESPARSE_MM_URL}/SNAP/web-Google.tar.gz"
    download_dataset TSOPF_RS_b2383 "${SUITESPARSE_MM_URL}/TSOPF/TSOPF_RS_b2383.tar.gz"
}

download_hipmcl() {
    echo "Downloading hipmcl dataset to ${OUTPUT_DIR}"
    if [ -f "${OUTPUT_DIR}/virus/vir_vs_vir_30_50length_propermm.mtx" ]; then
        echo "${OUTPUT_DIR}/virus dataset exists."
    else
        download_file "https://portal.nersc.gov/project/m1982/HipMCL/viruses/vir_vs_vir_30_50length_propermm.mtx" "${OUTPUT_DIR}/virus/vir_vs_vir_30_50length_propermm.mtx"
    fi
    if [ -f "${OUTPUT_DIR}/eukarya/euk_vs_euk_30_50length_propermm.mtx" ]; then
        echo "${OUTPUT_DIR}/eukarya dataset exists."
    else
        download_file "https://portal.nersc.gov/project/m1982/HipMCL/eukarya/euk_vs_euk_30_50length_propermm.mtx" "${OUTPUT_DIR}/eukarya/euk_vs_euk_30_50length_propermm.mtx"
    fi
    if [ -f "${OUTPUT_DIR}/archaea/arch_vs_arch_30_50length_propermm.mtx" ]; then
        echo "${OUTPUT_DIR}/archaea dataset exists."
    else
        download_file "https://portal.nersc.gov/project/m1982/HipMCL/archaea/arch_vs_arch_30_50length_propermm.mtx" "${OUTPUT_DIR}/archaea/arch_vs_arch_30_50length_propermm.mtx"
    fi
}

download_tfCombBLAS() {
    echo "Downloading dataset used for benchmarking CombBLAS GPU to ${OUTPUT_DIR}"
    download_dataset atmosmodd "${SUITESPARSE_MM_URL}/Bourchtein/atmosmodd.tar.gz"
    download_dataset delaunay_n22 "${SUITESPARSE_MM_URL}/DIMACS10/delaunay_n22.tar.gz"
}

download_ACSpGEMM() {
    echo "Downloading dataset used for benchmarking AC-SpGEMM Paper to ${OUTPUT_DIR}"
    download_dataset language "${SUITESPARSE_MM_URL}/Tromble/language.tar.gz"
    download_dataset scircuit "${SUITESPARSE_MM_URL}/Hamm/scircuit.tar.gz"
    download_dataset stat96v2 "${SUITESPARSE_MM_URL}/Meszaros/stat96v2.tar.gz"
    download_dataset asia_osm "${SUITESPARSE_MM_URL}/DIMACS10/asia_osm.tar.gz"
    download_dataset atmosmodl "${SUITESPARSE_MM_URL}/Bourchtein/atmosmodl.tar.gz"
    download_dataset filter3D "${SUITESPARSE_MM_URL}/Oberwolfach/filter3D.tar.gz"
    download_dataset bibd_19_9 "${SUITESPARSE_MM_URL}/JGD_BIBD/bibd_19_9.tar.gz"
    download_dataset cant "${SUITESPARSE_MM_URL}/Williams/cant.tar.gz"
    download_dataset landmark "${SUITESPARSE_MM_URL}/Pereyra/landmark.tar.gz"
    download_dataset hood "${SUITESPARSE_MM_URL}/GHS_psdef/hood.tar.gz"
    download_dataset TSOPF_RS_b2383 "${SUITESPARSE_MM_URL}/TSOPF/TSOPF_RS_b2383.tar.gz"
    download_dataset hugebubbles-00020 "${SUITESPARSE_MM_URL}/DIMACS10/hugebubbles-00020.tar.gz"
    download_dataset poisson3Da "${SUITESPARSE_MM_URL}/FEMLAB/poisson3Da.tar.gz"
    download_dataset TSC_OPF_1047 "${SUITESPARSE_MM_URL}/IPSO/TSC_OPF_1047.tar.gz"
}

download_suitsparse_small() {
    echo "Downloading SuiteSparse small dataset to ${OUTPUT_DIR}"
    download_dataset 1138_bus "${SUITESPARSE_MM_URL}/HB/1138_bus.tar.gz"
    download_dataset 494_bus "${SUITESPARSE_MM_URL}/HB/494_bus.tar.gz"
    download_dataset 662_bus "${SUITESPARSE_MM_URL}/HB/662_bus.tar.gz"
    download_dataset 685_bus "${SUITESPARSE_MM_URL}/HB/685_bus.tar.gz"
}

datasetname="${1:-}"

if [[ "$datasetname" == "tiny" ]]; then
    download_tiny
elif [[ "$datasetname" == "small" ]]; then
    download_small
elif [[ "$datasetname" == "medium" ]]; then
    download_medium
elif [[ "$datasetname" == "mcl" ]]; then
    download_hipmcl
elif [[ "$datasetname" == "tfcombblas" ]]; then
    download_tfCombBLAS
elif [[ "$datasetname" == "acspgemm" ]]; then
    download_ACSpGEMM
elif [[ "$datasetname" == "suitsparsesmall" ]]; then
    download_suitsparse_small
elif [[ -z "$datasetname" ]]; then
    echo "Usage: $0 [tiny|small|medium|mcl|tfcombblas|acspgemm|suitsparsesmall]"
else
    echo "Unknown dataset: $datasetname"
    echo "Usage: $0 [tiny|small|medium|mcl|tfcombblas|acspgemm|suitsparsesmall]"
fi
