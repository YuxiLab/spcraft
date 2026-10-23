#pragma once

#include <stdexcept>

#define make_sure(condition, message)       \
  do {                                     \
    if (!(condition)) {                    \
      throw std::runtime_error(message);   \
    }                                      \
  } while (false)
