#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>

// Follow the compiler by default; callers can force serial execution with 0.
#ifndef SPCRAFT_USE_OPENMP
#ifdef _OPENMP
#define SPCRAFT_USE_OPENMP 1
#else
#define SPCRAFT_USE_OPENMP 0
#endif
#endif

// clang-format off
#if defined(_OPENMP) && SPCRAFT_USE_OPENMP
  #define SPCRAFT_OPENMP_ENABLED 1
  #if defined(__clang__)
    // Workaround for clangd / clang: Clang does not support __malloc__ attribute with arguments,
    // which GCC 12 omp.h uses.
    #pragma push_macro("__malloc__")
    #undef __malloc__
    #define __malloc__(...)
  #endif
  #include <omp.h>
  #if defined(__clang__)
    #pragma pop_macro("__malloc__")
  #endif
  #define SPCRAFT_DO_PRAGMA(x)                 _Pragma(#x)
  #define OMP_PARALLEL(...)                    SPCRAFT_DO_PRAGMA(omp parallel __VA_ARGS__)
  #define OMP_FOR(...)                         SPCRAFT_DO_PRAGMA(omp for __VA_ARGS__)
  #define OMP_PARALLEL_FOR(...)                SPCRAFT_DO_PRAGMA(omp parallel for __VA_ARGS__)
  #define OMP_PARALLEL_FOR_STATIC(x)           SPCRAFT_DO_PRAGMA(omp parallel for schedule(static, x))
  #define OMP_PARALLEL_FOR_DYNAMIC(x)          SPCRAFT_DO_PRAGMA(omp parallel for schedule(dynamic, x))
  #define OMP_PARALLEL_FOR_REDUCTION_PLUS(var) SPCRAFT_DO_PRAGMA(omp parallel for reduction(+:var))
  #define OMP_SINGLE                           _Pragma("omp single")
  #define OMP_CRITICAL                         _Pragma("omp critical")
  #define OMP_FOR_STATIC(x)                    SPCRAFT_DO_PRAGMA(omp for schedule(static, x))
  #define OMP_FOR_DYNAMIC(x)                   SPCRAFT_DO_PRAGMA(omp for schedule(dynamic, x))
  #define OMP_BARRIER                          _Pragma("omp barrier")
  #define OMP_GET_MAX_THREADS()                omp_get_max_threads()
  #define OMP_GET_NUM_THREADS()                omp_get_num_threads()
  #define OMP_GET_THREAD_NUM()                 omp_get_thread_num()
  #define OMP_SET_NUM_THREADS(n)               omp_set_num_threads(n)
  #define OMP_SET_DYNAMIC(enabled)             omp_set_dynamic(enabled)
  #define OMP_GET_WTIME()                      omp_get_wtime()
  #define OMP_ATOMIC                           _Pragma("omp atomic")
  #define OMP_ATOMIC_CAPTURE                   _Pragma("omp atomic capture")
#else
  #define SPCRAFT_OPENMP_ENABLED 0
  #define OMP_PARALLEL(...)
  #define OMP_FOR(...)
  #define OMP_PARALLEL_FOR(...)
  #define OMP_PARALLEL_FOR_STATIC(x)
  #define OMP_PARALLEL_FOR_DYNAMIC(x)
  #define OMP_PARALLEL_FOR_REDUCTION_PLUS(var)
  #define OMP_SINGLE
  #define OMP_CRITICAL
  #define OMP_FOR_STATIC(x)
  #define OMP_FOR_DYNAMIC(x)
  #define OMP_BARRIER
  #define OMP_GET_MAX_THREADS()                1
  #define OMP_GET_NUM_THREADS()                1
  #define OMP_GET_THREAD_NUM()                 0
  #define OMP_SET_NUM_THREADS(n)               static_cast<void>(n)
  #define OMP_SET_DYNAMIC(enabled)             static_cast<void>(enabled)
  #define OMP_GET_WTIME()                      spcraft::detail::OmpSerialWallTime()
  #define OMP_ATOMIC
  #define OMP_ATOMIC_CAPTURE
#endif

#if SPCRAFT_OPENMP_ENABLED && __has_include(<parallel/algorithm>) && __has_include(<parallel/numeric>)
  #include <parallel/algorithm>
  #include <parallel/numeric>
  #define OMP_SORT                __gnu_parallel::sort
  #define OMP_STABLE_SORT         __gnu_parallel::stable_sort
  #define OMP_MAX_ELEMENT         __gnu_parallel::max_element
  #define OMP_MIN_ELEMENT         __gnu_parallel::min_element
  #define OMP_IS_SORTED           std::is_sorted
  #define OMP_ACCUMULATE          __gnu_parallel::accumulate
  #define OMP_TRANSFORM           __gnu_parallel::transform
  #define OMP_FIND                __gnu_parallel::find
  #define OMP_PARTIAL_SUM         __gnu_parallel::partial_sum
  #define OMP_INNER_PRODUCT       __gnu_parallel::inner_product
  #define OMP_ADJACENT_DIFFERENCE __gnu_parallel::adjacent_difference
  #define OMP_FOR_EACH            __gnu_parallel::for_each
  #define OMP_COUNT               __gnu_parallel::count
  #define OMP_COUNT_IF            __gnu_parallel::count_if
  #define OMP_REPLACE             __gnu_parallel::replace
  #define OMP_REPLACE_IF          __gnu_parallel::replace_if
#else
  #include <algorithm>
  #include <numeric>
  #define OMP_SORT                std::sort
  #define OMP_STABLE_SORT         std::stable_sort
  #define OMP_MAX_ELEMENT         std::max_element
  #define OMP_MIN_ELEMENT         std::min_element
  #define OMP_IS_SORTED           std::is_sorted
  #define OMP_ACCUMULATE          std::accumulate
  #define OMP_TRANSFORM           std::transform
  #define OMP_FIND                std::find
  #define OMP_PARTIAL_SUM         std::partial_sum
  #define OMP_INNER_PRODUCT       std::inner_product
  #define OMP_ADJACENT_DIFFERENCE std::adjacent_difference
  #define OMP_FOR_EACH            std::for_each
  #define OMP_COUNT               std::count
  #define OMP_COUNT_IF            std::count_if
  #define OMP_REPLACE             std::replace
  #define OMP_REPLACE_IF          std::replace_if
#endif
// clang-format on
