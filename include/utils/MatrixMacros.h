#pragma once

#include "core/CscMatrix.h"
#include "core/CsrMatrix.h"
#include "core/CooMatrix.h"

// Matrix shortcuts use the parameter names introduced by SP_MAT_TEMP.
#define SP_MAT_TEMP template <class IT, class NT, class OT>
#define SP_SR_MAT_TEMP template <class SemiRing, class IT, class NT, class OT>
#define SPCSC ::spcraft::CscMatrix<IT, NT, OT>
#define SPCSR ::spcraft::CsrMatrix<IT, NT, OT>
#define SPCOO ::spcraft::CooMatrix<IT, NT, OT>

#define SPND [[nodiscard]]
