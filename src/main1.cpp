#include "rank_predictor/dp.hpp"
#include <format>
#include <iostream>

int main()
{
  using rank_predictor::predict_rank;

  const auto rank = predict_rank({25000, 33000, 17000, 25000}, {30}, {3, 4, 5}, 0.2, 7, 0);

  for (std::size_t i = 0; i < rank.size(); ++i) {
    for (std::size_t j = 0; j < rank[i].size(); ++j) {
      std::cout << std::format("{:.4f} ", rank[i][j]);
    }

    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
