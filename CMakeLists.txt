cmake_minimum_required(VERSION 3.12)

project(rank-predictor LANGUAGES CXX)

include(GNUInstallDirs)

add_library(rank_predictor
  src/detail/classify.cpp
  src/detail/score.cpp
  src/common.cpp
  src/dp.cpp
  src/mc.cpp
  src/nd.cpp)
target_compile_features(rank_predictor PUBLIC cxx_std_23)
target_compile_options(rank_predictor PUBLIC
  $<$<CONFIG:Debug>:-O0 -Wall -Wextra -g>
  $<$<CONFIG:Release>:-O3 -Wall -Wextra>)
target_include_directories(rank_predictor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS rank_predictor
  EXPORT RankPredictor
  LIBRARY DESTINATION lib)
install(EXPORT RankPredictor
  FILE RankPredictorConfig.cmake
  NAMESPACE RankPredictor::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/share/cmake)

add_executable(main1 tests/main1.cpp)
target_link_libraries(main1 rank_predictor)

add_executable(main2 tests/main2.cpp)
target_link_libraries(main2 rank_predictor)

add_executable(main3 tests/main3.cpp)
target_link_libraries(main3 rank_predictor)
