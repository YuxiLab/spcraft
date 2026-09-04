#pragma once

namespace spcraft
{

#if defined(__CUDACC__)
#define SPCRAFT_SEMIRING_HOST_DEVICE __host__ __device__
#else
#define SPCRAFT_SEMIRING_HOST_DEVICE
#endif

/**
 * @brief Semiring using addition and multiplication.
 *
 * With bool values, conversion of addition back to bool behaves as logical OR
 * and multiplication behaves as logical AND, giving the OR-AND semiring used
 * for BFS frontier expansion.
 */
template <class NT>
struct PlusTimesRing {
  using ValueType = NT;

  inline static constexpr NT kAdditiveIdentity = NT{0};
  inline static constexpr NT kMultiplicativeIdentity = NT{1};

  [[nodiscard]] SPCRAFT_SEMIRING_HOST_DEVICE static constexpr NT Add(const NT& left,
                                                                     const NT& right)
  {
    return left + right;
  }

  [[nodiscard]] SPCRAFT_SEMIRING_HOST_DEVICE static constexpr NT Multiply(const NT& left,
                                                                          const NT& right)
  {
    return left * right;
  }
};

#undef SPCRAFT_SEMIRING_HOST_DEVICE

}  // namespace spcraft
