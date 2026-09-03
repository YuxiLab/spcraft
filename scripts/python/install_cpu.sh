#!/usr/bin/env bash

set -euo pipefail

printf 'Installing spcraft Python package\n'
printf '  MPI:  OFF\n'
printf '  CUDA: OFF\n\n'

uv sync --verbose --reinstall-package spcraft \
    -Clogging.level=INFO \
    -Ccmake.define.SPCRAFT_USE_MPI=OFF \
    -Ccmake.define.SPCRAFT_USE_CUDA=OFF
