#!/usr/bin/env bash

set -euo pipefail

printf 'Installing spcraft Python package\n'
printf '  MPI:  ON\n'
printf '  CUDA: OFF\n\n'

uv sync --verbose --reinstall-package spcraft \
    -Clogging.level=INFO \
    -Ccmake.define.SPCRAFT_USE_MPI=ON \
    -Ccmake.define.SPCRAFT_USE_CUDA=OFF
