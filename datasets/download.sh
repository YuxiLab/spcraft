#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${DLOC:-${OUTPUT_DIR:-${SCRIPT_DIR}/output}}"
mkdir -p "${OUTPUT_DIR}"

# Helper to download a file using curl or wget
download_file() {
    local url="$1"
    local dest_file="$2"
    mkdir -p "$(dirname "${dest_file}")"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL -o "${dest_file}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --no-check-certificate -O "${dest_file}" "${url}"
    else
        echo "Error: neither curl nor wget found."
        return 1
    fi
}

# Function to download a dataset
download_dataset() {
    local dataset_name="$1"
    local url="$2"

    if [ -d "${OUTPUT_DIR}/${dataset_name}" ]; then
        echo "${OUTPUT_DIR}/${dataset_name} dataset exists."
    else
        echo "Downloading ${dataset_name} to ${OUTPUT_DIR}..."
        rm -f "${OUTPUT_DIR}/${dataset_name}.tar.gz"
        download_file "${url}" "${OUTPUT_DIR}/${dataset_name}.tar.gz"
        tar -xzf "${OUTPUT_DIR}/${dataset_name}.tar.gz" -C "${OUTPUT_DIR}"
        rm -f "${OUTPUT_DIR}/${dataset_name}.tar.gz"
    fi
}

download_tiny() {
    echo "Downloading tiny dataset to ${OUTPUT_DIR}"
    # tilespgemm 
    download_dataset pdb1HYS https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
    download_dataset consph https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz
    download_dataset webbase-1M https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
    download_dataset bcspwr06 https://suitesparse-collection-website.herokuapp.com/MM/HB/bcspwr06.tar.gz
}

download_small() {
    echo "Downloading small dataset to ${OUTPUT_DIR}"
    # tilespgemm 
    download_dataset pdb1HYS https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
    download_dataset consph https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz
    download_dataset webbase-1M https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
    download_dataset af_shell10 https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz
    download_dataset SiO2 https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/SiO2.tar.gz
    download_dataset gupta3 https://suitesparse-collection-website.herokuapp.com/MM/Gupta/gupta3.tar.gz
    download_dataset rma10 https://suitesparse-collection-website.herokuapp.com/MM/Bova/rma10.tar.gz
    download_dataset TSOPF_FS_b300_c2 https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_FS_b300_c2.tar.gz
}

download_medium() {
    echo "Downloading medium dataset to ${OUTPUT_DIR}"
    # SpGEMM/SpMV medium benchmark matrices (~50MB-300MB uncompressed / ~10MB-120MB compressed)
    download_dataset cop20k_A https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz
    download_dataset cant https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz
    download_dataset mac_econ_fwd500 https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz
    download_dataset pwtk https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz
    download_dataset hood https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/hood.tar.gz
    download_dataset bmwcra_1 https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/bmwcra_1.tar.gz
    download_dataset web-Google https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz
    download_dataset TSOPF_RS_b2383 https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_RS_b2383.tar.gz
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
    download_dataset atmosmodd https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodd.tar.gz
    download_dataset delaunay_n22 https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n22.tar.gz 
}

download_ACSpGEMM() {
    echo "Downloading dataset used for benchmarking AC-SpGEMM Paper to ${OUTPUT_DIR}"
    download_dataset language https://suitesparse-collection-website.herokuapp.com/MM/Tromble/language.tar.gz
    download_dataset scircuit https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz
    download_dataset stat96v2 https://suitesparse-collection-website.herokuapp.com/MM/Meszaros/stat96v2.tar.gz
    download_dataset asia_osm https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/asia_osm.tar.gz
    download_dataset atmosmodl https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodl.tar.gz
    download_dataset filter3D https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/filter3D.tar.gz
    download_dataset bibd_19_9 https://suitesparse-collection-website.herokuapp.com/MM/JGD_BIBD/bibd_19_9.tar.gz
    download_dataset cant https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz
    download_dataset landmark https://suitesparse-collection-website.herokuapp.com/MM/Pereyra/landmark.tar.gz
    download_dataset hood https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/hood.tar.gz
    download_dataset TSOPF_RS_b2383 https://suitesparse-collection-website.herokuapp.com/MM/TSOPF/TSOPF_RS_b2383.tar.gz
    download_dataset hugebubbles-00020 https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/hugebubbles-00020.tar.gz
    download_dataset poisson3Da https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Da.tar.gz
    download_dataset TSC_OPF_1047 https://suitesparse-collection-website.herokuapp.com/MM/IPSO/TSC_OPF_1047.tar.gz
}

download_suitsparse_small() {
    echo "Downloading SuiteSparse small dataset to ${OUTPUT_DIR}"
    download_dataset 1138_bus https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
    download_dataset 494_bus https://suitesparse-collection-website.herokuapp.com/MM/HB/494_bus.tar.gz
    download_dataset 662_bus https://suitesparse-collection-website.herokuapp.com/MM/HB/662_bus.tar.gz
    download_dataset 685_bus https://suitesparse-collection-website.herokuapp.com/MM/HB/685_bus.tar.gz
}

datasetname="$1"

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