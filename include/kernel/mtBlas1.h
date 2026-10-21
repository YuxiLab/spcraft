#pragma once
#include <stdexcept>

#include "utils/utils.h"
#include "utils/MatrixMacros.h"

namespace spcraft
{
namespace detail
{

//! Dot Product of two dense vectors, return a scalar.
template <class IT, class NT>
[[nodiscard]] NT Dot(const SPDVEC& left, const SPDVEC& right)
{
  const IT n = left.n;
  NT total = NT{0};
  OMP_PARALLEL_FOR(schedule(static) reduction(+ : total))
  for (IT index = 0; index < n; ++index) {
    total += left.val[index] * right.val[index];
  }
  return total;
}

//! Scale a dense vector: x <- alpha * x.
template <class IT, class NT>
void Scale(NT alpha, SPDVEC& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = alpha * x.val[index];
  }
}

//! y <- y + alpha * x.
template <class IT, class NT>
void Axpy(NT alpha, const SPDVEC& x, SPDVEC& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT index = 0; index < n; ++index) {
    y.val[index] += alpha * x.val[index];
  }
}

//! y <- x + beta * y.
template <class IT, class NT>
void Aypx(NT beta, const SPDVEC& x, SPDVEC& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index] + beta * y.val[index];
  }
}

//! x <- value, everywhere.
template <class IT, class NT>
void Fill(NT value, SPDVEC& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = value;
  }
}

//! copy vectors
template <class IT, class NT>
void Copy(const SPDVEC& x, SPDVEC& y)
{
  if (x.n != y.n) throw std::runtime_error("vector dimension not matched!");
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index];
  }
}

//! Backward-compatible wrappers
template <class IT, class NT>
[[nodiscard]] NT DotOpenMP(const SPDVEC& left, const SPDVEC& right) { return Dot(left, right); }

template <class IT, class NT>
void AxpyOpenMP(NT alpha, const SPDVEC& x, SPDVEC& y) { Axpy(alpha, x, y); }

template <class IT, class NT>
void AypxOpenMP(NT beta, const SPDVEC& x, SPDVEC& y) { Aypx(beta, x, y); }

template <class IT, class NT>
void ScaleOpenMP(NT alpha, SPDVEC& x) { Scale(alpha, x); }

template <class IT, class NT>
void FillOpenMP(NT value, SPDVEC& x) { Fill(value, x); }

}  // namespace detail
}  // namespace spcraft
