cmake_minimum_required(VERSION 3.5)

project(rank-predictor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

add_library(rank_predictor STATIC
  src/rank_predictor/detail/classify.cpp
  src/rank_predictor/detail/score.cpp
  src/rank_predictor/common.cpp
  src/rank_predictor/dp.cpp
  src/rank_predictor/mc.cpp)
set(PUBLIC_HEADERS
  src/rank_predictor/dp.hpp
  src/rank_predictor/mc.hpp)
set_target_properties(rank_predictor PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
install(TARGETS rank_predictor EXPORT RankPredictor
  PUBLIC_HEADER DESTINATION include/rank_predictor)
install(EXPORT RankPredictor FILE RankPredictorConfig.cmake DESTINATION share/cmake)

add_executable(main1 src/main1.cpp)
target_link_libraries(main1 rank_predictor)

add_executable(main2 src/main2.cpp)
target_link_libraries(main2 rank_predictor)
