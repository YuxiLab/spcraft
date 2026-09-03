#pragma once

#include "CooMatrix.h"
#include "CscMatrix.h"
#include "CsrMatrix.h"
#include "DenseVector.h"
#include "mtSpMV.h"

#include "MatrixGenerator.h"
#include "SemiRing.h"

#ifdef SPCRAFT_USE_MKL
#include "mklSpMV.h"
#endif

#ifdef SPCRAFT_USE_CUDA
#include "cuSpMV.cuh"
#endif
