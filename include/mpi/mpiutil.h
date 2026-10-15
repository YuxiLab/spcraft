#pragma once

#include <mpi.h>

#include <cstdint>
#include <type_traits>

namespace spcraft
{

/**
 * @brief Map fixed-width integer, float, double, and bool types to predefined MPI datatypes.
 * The returned handle is borrowed: do not commit or free it. CV-qualified scalar
 * types use the same mapping. Unsupported types are rejected at compile time.
 */
template <class T>
[[nodiscard]] MPI_Datatype MPIType()
{
  if constexpr (std::is_const_v<T> || std::is_volatile_v<T>) {
    return MPIType<std::remove_cv_t<T>>();
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, bool>) {
    return MPI_CXX_BOOL;
  } else {
    static_assert(!std::is_same_v<T, T>, "MPIType requires a supported scalar type");
  }
}

}  // namespace spcraft
