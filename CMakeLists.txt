cmake_minimum_required(VERSION 3.5)

project(rank-predictor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

add_library(rank_predictor STATIC
  src/rank_predictor/detail/classify.cpp
  src/rank_predictor/detail/score.cpp
  src/rank_predictor/dp.cpp)
set_target_properties(rank_predictor PROPERTIES PUBLIC_HEADER src/rank_predictor.hpp)
install(TARGETS rank_predictor EXPORT RankPredictor)
install(EXPORT RankPredictor FILE RankPredicorConfig.cmake DESTINATION share/cmake)

add_executable(main src/main.cpp)
target_compile_definitions(main PUBLIC INDEX_FILE_PATH="${PROJECT_BINARY_DIR}")
target_link_libraries(main rank_predictor)
