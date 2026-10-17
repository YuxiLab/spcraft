#pragma once

#include "utils/omp/omp_wrapper.h"
#include "utils/MatrixInput.h"
#include "utils/MatrixMacros.h"
#include "utils/Print.h"
#include "utils/StdAliases.h"

#include "core/CooMatrix.h"
#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/DcscMatrix.h"
#include "core/DenseVector.h"
#include "core/SparseVector.h"
#include "eigen/eigeninterface.h"
#include "graph/PageRank.h"
#include "graph/mtPageRank.h"
#include "kernel/mtBlas1.h"
#include "kernel/mtSDDMM.h"
#include "kernel/mtSpGEMM.h"
#include "kernel/mtSpMM.h"
#include "kernel/mtSpMV.h"
#include "linear_algebra/mtConjugateGradient.h"
#include "linear_algebra/mtPowerIteration.h"
#include "numeric/mtMultiwayMerge.h"

#include "matrix_generator/MatrixGenerator.h"
#include "report/BenchmarkReport.h"
#include "semiring/SemiRing.h"

#ifdef SPCRAFT_USE_MKL
#include "kernel/mklSpMV.h"
#endif

#ifdef SPCRAFT_USE_CUDA
#include "core/cuCooMatrix.cuh"
#include "core/cuCscMatrix.cuh"
#include "core/cuCsrMatrix.cuh"
#include "graph/cuPageRank.cuh"
#include "kernel/cuSpMV.cuh"
#endif

#ifdef SPCRAFT_USE_MPI
#include "mpi/mpiutil.h"
#include "mpi/CommGrid2D.h"
#include "mpi/CommGrid3D.h"
#include "mpi/3dspgemm.h"
#include "mpi/DistSpCOO.h"
#include "mpi/DistSpCSR.h"
#include "mpi/DistSpCSC.h"
#include "mpi/DistSpDCSC.h"
#include "mpi/2dspgemm.h"
#endif
