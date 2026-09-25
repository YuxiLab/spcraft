#pragma once
#include <stdexcept>

#include "utils/utils.h"

namespace spcraft
{
namespace detail
{

//! Dot Product of two dense vectors, return a scalar.
SP_VEC_TEMP
SPND NT Dot(const SPDVEC& left, const SPDVEC& right)
{
  const IT n = left.n;
  NT total = NT{0};
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(left, right, n) reduction(+ : total))
  for (IT index = 0; index < n; ++index) {
    total += left.val[index] * right.val[index];
  }
  return total;
}

//! Scale a dense vector: x <- alpha * x.
SP_VEC_TEMP
void Scale(NT alpha, SPDVEC& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, n, alpha))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = alpha * x.val[index];
  }
}

//! y <- y + alpha * x.
SP_VEC_TEMP
void Axpy(NT alpha, const SPDVEC& x, SPDVEC& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n, alpha))
  for (IT index = 0; index < n; ++index) {
    y.val[index] += alpha * x.val[index];
  }
}

//! y <- x + beta * y.
SP_VEC_TEMP
void Aypx(NT beta, const SPDVEC& x, SPDVEC& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n, beta))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index] + beta * y.val[index];
  }
}

//! x <- value, everywhere.
SP_VEC_TEMP
void Fill(NT value, SPDVEC& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, n, value))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = value;
  }
}

//! copy vectors
SP_VEC_TEMP
void Copy(const SPDVEC& x, SPDVEC& y)
{
  if (x.n != y.n) throw std::runtime_error("vector dimension not matched!");
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index];
  }
}

//! Backward-compatible wrappers
SP_VEC_TEMP
SPND NT DotOpenMP(const SPDVEC& left, const SPDVEC& right) { return Dot(left, right); }

SP_VEC_TEMP
void AxpyOpenMP(NT alpha, const SPDVEC& x, SPDVEC& y) { Axpy(alpha, x, y); }

SP_VEC_TEMP
void AypxOpenMP(NT beta, const SPDVEC& x, SPDVEC& y) { Aypx(beta, x, y); }

SP_VEC_TEMP
void ScaleOpenMP(NT alpha, SPDVEC& x) { Scale(alpha, x); }

SP_VEC_TEMP
void FillOpenMP(NT value, SPDVEC& x) { Fill(value, x); }

}  // namespace detail
}  // namespace spcraft
