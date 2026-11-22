cmake_minimum_required(VERSION 3.20)

project(rank-predictor LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Boost 1.81 REQUIRED)

add_library(rank_predictor
  src/detail/classify.cpp
  src/detail/score.cpp
  src/common.cpp
  src/mc.cpp
  src/nd.cpp)
target_link_libraries(rank_predictor PRIVATE
  $<BUILD_INTERFACE:Boost::headers>)
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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/mahjong DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT RankPredictor
  FILE RankPredictorConfig.cmake
  NAMESPACE RankPredictor::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mahjong)

add_executable(main2 tests/main2.cpp)
target_link_libraries(main2 rank_predictor)

add_executable(main3 tests/main3.cpp)
target_link_libraries(main3 rank_predictor)
