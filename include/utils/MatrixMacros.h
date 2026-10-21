#pragma once

// Matrix shortcuts require the corresponding type to be declared or included.
#define SPCSC CscMatrix<IT, NT, OT>
#define SPDCSC DcscMatrix<IT, NT, OT>
#define SPCSR CsrMatrix<IT, NT, OT>
#define SPCOO CooMatrix<IT, NT, OT>
#define DISTSPCOO DistSpCOO<IT, NT, OT>
#define DISTSPCSR DistSpCSR<IT, NT, OT>
#define DISTSPCSC DistSpCSC<IT, NT, OT>
#define DISTSPDCSC DistSpDCSC<IT, NT, OT>
#define SPDVEC DenseVector<IT, NT>
#define SPSVEC SparseVector<IT, NT>

// CHECK
#define CHECK_INTTYPE_LIMIT(val, T)                                                                      \
  do {                                                                                                   \
    if (static_cast<std::uintmax_t>(val) > static_cast<std::uintmax_t>(std::numeric_limits<T>::max())) { \
      std::string err_msg =                                                                              \
          std::string("Inttype exceed limits at ") + __FILE__ ":" + std::to_string(__LINE__);            \
      throw std::overflow_error(err_msg);                                                                \
    }                                                                                                    \
  } while (0)
