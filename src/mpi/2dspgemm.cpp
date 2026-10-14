#include "mpi/2dspgemm.h"

#include <cstdint>
#include <type_traits>

#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include "semiring/SemiRing.h"

namespace spcraft
{

SP_SR_MAT_TEMP
SPND SPCOO Mpi2DSummaSpGEMM(const SPDCSC& A, const SPDCSC& B, MPI_Comm grid)
{
  static_assert(std::is_same_v<typename SemiRing::ValueType, NT>,
                "Semiring value type must match matrix value type");
  static_cast<void>(grid);

  // TODO: Implement the collective SUMMA stages. A shaped empty result lets
  // students run the tests and see numerical failures before filling them in.
  SPCOO result;
  result.Allocate(OT{0}, A.m, B.n);
  return result;
}

#define SPCRAFT_DEFINE_SUMMA_2D_TYPES(IT, NT, OT)                                 \
  template CooMatrix<IT, NT, OT> Mpi2DSummaSpGEMM<PlusTimesRing<NT>, IT, NT, OT>( \
      const DcscMatrix<IT, NT, OT>&, const DcscMatrix<IT, NT, OT>&, MPI_Comm);

#define SPCRAFT_DEFINE_SUMMA_2D(r, product)                                          \
  SPCRAFT_DEFINE_SUMMA_2D_TYPES(BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(0, product)), \
                                BOOST_PP_SEQ_ELEM(1, product),                       \
                                BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(0, product)))

// Keep the exercise's compiled storage pairs consistent with local SpGEMM.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(SPCRAFT_DEFINE_SUMMA_2D,
                              (((std::int32_t)(std::int32_t))
                               ((std::int32_t)(std::int64_t))
                               ((std::int64_t)(std::int64_t)))
                              ((float)(double)))
// clang-format on
#undef SPCRAFT_DEFINE_SUMMA_2D
#undef SPCRAFT_DEFINE_SUMMA_2D_TYPES

}  // namespace spcraft
