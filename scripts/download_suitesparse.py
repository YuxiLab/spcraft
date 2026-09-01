#!/usr/bin/env python3
"""
Download sparse matrices from the SuiteSparse Matrix Collection (https://sparse.tamu.edu).
Includes curated matrix suites widely used in SpGEMM and SpMV research paper benchmarks.
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Popular matrices used in SpGEMM and SpMV research papers
SPGEMM_BENCHMARKS = {
    "1138_bus": "HB",
    "bcsstk17": "HB",
    "cop20k_A": "Williams",
    "consph": "Williams",
    "amazon0302": "SNAP",
    "web-Google": "SNAP",
    "scircuit": "Hamm",
    "cnr-2000": "LAW",
}

SPMV_BENCHMARKS = {
    "pwtk": "Boeing",
    "inline_1": "GHS_psdef",
    "ct20stif": "Boeing",
    "cavity04": "DRIVCAV",
    "bmwcra_1": "GHS_psdef",
    "Flan_1565": "Janna",
}

ALL_PRESETS = {**SPGEMM_BENCHMARKS, **SPMV_BENCHMARKS}
BASE_URL = "https://sparse.tamu.edu/MM"


def download_matrix(name_or_path: str, output_dir: Path) -> bool:
    """Download and extract a matrix from SuiteSparse Matrix Collection."""
    if "/" in name_or_path:
        group, name = name_or_path.split("/", 1)
    else:
        name = name_or_path
        group = ALL_PRESETS.get(name)
        if not group:
            print(
                f"Error: Group for matrix '{name}' is unknown. Please specify as 'Group/MatrixName' (e.g. Williams/cop20k_A)."
            )
            return False

    if name.endswith(".mtx"):
        name = name[:-4]
    if name.endswith(".tar.gz"):
        name = name[:-7]

    url = f"{BASE_URL}/{group}/{name}.tar.gz"
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{name}.tar.gz"
    mtx_target = output_dir / f"{name}.mtx"

    if mtx_target.exists():
        print(f"Matrix '{name}.mtx' already exists in {output_dir}. Skipping download.")
        return True

    print(f"Downloading '{group}/{name}' from {url}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response, open(tar_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if tar_path.exists():
            tar_path.unlink()
        return False

    print(f"Extracting '{name}.mtx' to {output_dir}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            mtx_member = None
            for member in tar.getmembers():
                if member.name.endswith(".mtx"):
                    mtx_member = member
                    break

            if not mtx_member:
                print(f"Error: No .mtx file found inside {tar_path}")
                tar_path.unlink()
                return False

            f = tar.extractfile(mtx_member)
            if f is None:
                print(f"Error: Could not extract {mtx_member.name}")
                tar_path.unlink()
                return False

            with open(mtx_target, "wb") as out_mtx:
                shutil.copyfileobj(f, out_mtx)

        print(f"Successfully saved to: {mtx_target}")
        return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False
    finally:
        if tar_path.exists():
            tar_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Download sparse matrices from SuiteSparse Matrix Collection for SpGEMM/SpMV paper benchmarks."
    )
    parser.add_argument(
        "-m",
        "--matrix",
        nargs="+",
        help="Matrix name(s) to download (e.g. cop20k_A or Williams/cop20k_A)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data",
        help="Output directory to save .mtx files (default: data)",
    )
    parser.add_argument(
        "--spgemm",
        action="store_true",
        help="Download standard SpGEMM paper benchmark matrices",
    )
    parser.add_argument(
        "--spmv",
        action="store_true",
        help="Download standard SpMV paper benchmark matrices",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Download all pre-configured paper benchmark matrices",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List pre-configured paper benchmark matrices",
    )

    args = parser.parse_args()

    if args.list:
        print("=== SpGEMM Paper Benchmarks ===")
        for name, group in SPGEMM_BENCHMARKS.items():
            print(f"  - {group}/{name}")
        print("\n=== SpMV Paper Benchmarks ===")
        for name, group in SPMV_BENCHMARKS.items():
            print(f"  - {group}/{name}")
        return

    matrices = []
    if args.spgemm:
        matrices.extend([f"{g}/{n}" for n, g in SPGEMM_BENCHMARKS.items()])
    if args.spmv:
        matrices.extend([f"{g}/{n}" for n, g in SPMV_BENCHMARKS.items()])
    if args.all:
        matrices.extend([f"{g}/{n}" for n, g in ALL_PRESETS.items()])
    if args.matrix:
        matrices.extend(args.matrix)

    if not matrices:
        print("No matrix specified. Downloading standard SpGEMM paper benchmark matrices...")
        matrices = [f"{g}/{n}" for n, g in SPGEMM_BENCHMARKS.items()]

    seen = set()
    dedup_matrices = []
    for m in matrices:
        if m not in seen:
            seen.add(m)
            dedup_matrices.append(m)

    output_dir = Path(args.output_dir)
    success_count = 0
    for mat in dedup_matrices:
        if download_matrix(mat, output_dir):
            success_count += 1

    print(f"\nDone: Downloaded/verified {success_count}/{len(dedup_matrices)} matrices in '{output_dir}'.")


if __name__ == "__main__":
    main()
