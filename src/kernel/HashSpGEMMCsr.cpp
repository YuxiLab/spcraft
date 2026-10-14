#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include "kernel/mtSpGEMM.h"
#include "semiring/SemiRing.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{

SP_SR_MAT_TEMP
SPND SPCOO OmpHashSpGEMM(const SPCSR& A, const SPCSR& B)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  if (A.n != B.m) throw std::invalid_argument("SpGEMM matrix dimensions do not match");
  if (A.m < 0 || A.n < 0 || B.m < 0 || B.n < 0)
    throw std::invalid_argument("SpGEMM dimensions must be nonnegative");
  SPCOO result;
  if (!A.nnz || !B.nnz) {
    result.Allocate(0, A.m, B.n);
    return result;
  }
  return result;
}

#define SPCRAFT_DEFINE_HASH_SPGEMM_CSR_TYPES(IT, NT, OT)                       \
  template CooMatrix<IT, NT, OT> OmpHashSpGEMM<PlusTimesRing<NT>, IT, NT, OT>( \
      const CsrMatrix<IT, NT, OT>&, const CsrMatrix<IT, NT, OT>&);

#define SPCRAFT_DEFINE_HASH_SPGEMM_CSR(r, product)                                          \
  SPCRAFT_DEFINE_HASH_SPGEMM_CSR_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                       BOOST_PP_SEQ_ELEM(1, product),                       \
                                       BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Pair the storage types so 64-bit indices never select 32-bit offsets.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_HASH_SPGEMM_CSR,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_HASH_SPGEMM_CSR
#undef SPCRAFT_DEFINE_HASH_SPGEMM_CSR_TYPES

}  // namespace spcraft
