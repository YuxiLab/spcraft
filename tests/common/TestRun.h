#pragma once

#include <gtest/gtest.h>

namespace spcraft::tests
{
inline int TestRun(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
}  // namespace spcraft::tests
