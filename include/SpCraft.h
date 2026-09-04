#pragma once

#include "core/CooMatrix.h"
#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/DenseVector.h"
#include "kernel/mtSDDMM.h"
#include "kernel/mtSpGEMM.h"
#include "kernel/mtSpMM.h"
#include "kernel/mtSpMV.h"

#include "matrix_generator/MatrixGenerator.h"
#include "semiring/SemiRing.h"

#ifdef SPCRAFT_USE_MKL
#include "kernel/mklSpMV.h"
#endif

#ifdef SPCRAFT_USE_CUDA
#include "kernel/cuSpMV.cuh"
#endif
