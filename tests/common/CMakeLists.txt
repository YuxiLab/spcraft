add_library(SPCraftTestMain STATIC TestMain.cpp)
target_link_libraries(SPCraftTestMain PUBLIC GTest::gtest)
target_include_directories(SPCraftTestMain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
