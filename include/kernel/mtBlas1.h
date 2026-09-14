#pragma once

#include <stdexcept>
#include "utils/omp/omp_wrapper.h"

#include "core/DenseVector.h"

namespace spcraft
{
namespace detail
{

//! Dot Product of two dense vectors, return a scalar.
template <class IT, class NT>
[[nodiscard]] NT Dot(const DenseVector<IT, NT>& left, const DenseVector<IT, NT>& right)
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
template <class IT, class NT>
void Scale(NT alpha, DenseVector<IT, NT>& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, n, alpha))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = alpha * x.val[index];
  }
}

//! y <- y + alpha * x.
template <class IT, class NT>
void Axpy(NT alpha, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n, alpha))
  for (IT index = 0; index < n; ++index) {
    y.val[index] += alpha * x.val[index];
  }
}

//! y <- x + beta * y.
template <class IT, class NT>
void Aypx(NT beta, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  const IT n = y.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n, beta))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index] + beta * y.val[index];
  }
}

//! x <- value, everywhere.
template <class IT, class NT>
void Fill(NT value, DenseVector<IT, NT>& x)
{
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, n, value))
  for (IT index = 0; index < n; ++index) {
    x.val[index] = value;
  }
}

//! copy vectors
template <class IT, class NT>
void Copy(const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  if (x.n != y.n) throw std::runtime_error("vector dimension not matched!");
  const IT n = x.n;
  OMP_PARALLEL_FOR(schedule(static) default(none) shared(x, y, n))
  for (IT index = 0; index < n; ++index) {
    y.val[index] = x.val[index];
  }
}

//! Backward-compatible wrappers
template <class IT, class NT>
[[nodiscard]] NT DotOpenMP(const DenseVector<IT, NT>& left, const DenseVector<IT, NT>& right)
{
  return Dot(left, right);
}

template <class IT, class NT>
void AxpyOpenMP(NT alpha, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  Axpy(alpha, x, y);
}

template <class IT, class NT>
void AypxOpenMP(NT beta, const DenseVector<IT, NT>& x, DenseVector<IT, NT>& y)
{
  Aypx(beta, x, y);
}

template <class IT, class NT>
void ScaleOpenMP(NT alpha, DenseVector<IT, NT>& x)
{
  Scale(alpha, x);
}

template <class IT, class NT>
void FillOpenMP(NT value, DenseVector<IT, NT>& x)
{
  Fill(value, x);
}

}  // namespace detail
}  // namespace spcraft
