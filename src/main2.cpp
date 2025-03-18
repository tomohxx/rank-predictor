#include "rank_predictor/mc.hpp"
#include <format>
#include <iostream>
using namespace rank_predictor;

int main()
{
  Dist dist = {25000, 33000, 17000, 25000};

  std::random_device seed_gen;
  std::mt19937_64 engine(seed_gen());

  const auto rank = predict_rank(dist, {30}, {3, 4, 5}, 0.2, 7, 0, 10000, engine);

  for (int i = 0; i < NUM_PLAYERS; ++i) {
    for (int j = 0; j < NUM_PLAYERS; ++j) {
      std::cout << std::format("{:.4f} ", rank[i][j]);
    }

    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
