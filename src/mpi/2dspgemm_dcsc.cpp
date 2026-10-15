#include "mpi/2dspgemm.h"

#include <type_traits>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "semiring/SemiRing.h"
#include "SummaCommon.h"

namespace spcraft
{
SP_SR_MAT_TEMP
SPND DISTSPCOO Mpi2DSummaSpGEMM(const DISTSPDCSC& A, const DISTSPDCSC& B)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  DISTSPCOO result = detail::RunSummaStages<SemiRing, IT, NT, OT>(A, B, 0, 1);
  return result;
}

#define SPCRAFT_DEFINE_SUMMA_2D_DCSC_TYPES(IT, NT, OT)                            \
  template DistSpCOO<IT, NT, OT> Mpi2DSummaSpGEMM<PlusTimesRing<NT>, IT, NT, OT>( \
      const DistSpDCSC<IT, NT, OT>&, const DistSpDCSC<IT, NT, OT>&);

#define SPCRAFT_DEFINE_SUMMA_2D_DCSC(r, product)                                          \
  SPCRAFT_DEFINE_SUMMA_2D_DCSC_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                     BOOST_PP_SEQ_ELEM(1, product),                       \
                                     BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Keep compiled storage pairs consistent with local SpGEMM.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SUMMA_2D_DCSC,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_SUMMA_2D_DCSC
#undef SPCRAFT_DEFINE_SUMMA_2D_DCSC_TYPES

}  // namespace spcraft
