#pragma once

#include "CooMatrix.h"
#include "CscMatrix.h"
#include "CsrMatrix.h"
#include "mtSpMV.h"

#ifdef SPCRAFT_USE_MKL
#include "mklSpMV.h"
#endif

#ifdef SPCRAFT_USE_CUDA
#include "cuSpMV.cuh"
#endif
