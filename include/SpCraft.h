#pragma once

#include "utils/omp/omp_wrapper.h"

#include "core/CooMatrix.h"
#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/DcscMatrix.h"
#include "core/DenseVector.h"
#include "graph/PageRank.h"
#include "graph/mtPageRank.h"
#include "kernel/mtBlas1.h"
#include "kernel/mtSDDMM.h"
#include "kernel/mtSpGEMM.h"
#include "kernel/mtSpMM.h"
#include "kernel/mtSpMV.h"
#include "linear_algebra/mtConjugateGradient.h"
#include "linear_algebra/mtPowerIteration.h"

#include "matrix_generator/MatrixGenerator.h"
#include "report/BenchmarkReport.h"
#include "semiring/SemiRing.h"

#ifdef SPCRAFT_USE_MKL
#include "kernel/mklSpMV.h"
#endif

#ifdef SPCRAFT_USE_CUDA
#include "core/cuCsrMatrix.cuh"
#include "graph/cuPageRank.cuh"
#include "kernel/cuSpMV.cuh"
#endif
