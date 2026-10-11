#pragma once

// Matrix shortcuts use the parameter names introduced by SP_MAT_TEMP.
#define SP_VEC_TEMP template <class IT, class NT>
#define SP_MAT_TEMP template <class IT, class NT, class OT>
#define SP_SR_MAT_TEMP template <class SemiRing, class IT, class NT, class OT>
#define SPCSC CscMatrix<IT, NT, OT>
#define SPDCSC DcscMatrix<IT, NT, OT>
#define SPCSR CsrMatrix<IT, NT, OT>
#define SPCOO CooMatrix<IT, NT, OT>
#define SPDVEC DenseVector<IT, NT>
#define SPND [[nodiscard]]

#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/CooMatrix.h"
#include "core/DcscMatrix.h"
#include "core/DenseVector.h"

// CHECK
#define CHECK_INTTYPE_LIMIT(val, T)                                                           \
  do {                                                                                        \
    if (static_cast<std::uintmax_t>(val) >                                                    \
        static_cast<std::uintmax_t>(std::numeric_limits<T>::max())) {                         \
      std::string err_msg =                                                                   \
          std::string("Inttype exceed limits at ") + __FILE__ ":" + std::to_string(__LINE__); \
      throw std::overflow_error(err_msg);                                                     \
    }                                                                                         \
  } while (0)
