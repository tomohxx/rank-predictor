#ifndef RANK_PREDICTOR_HPP
#define RANK_PREDICTOR_HPP

#include <boost/container_hash/hash.hpp>
#include <unordered_map>
#include <valarray>
#include <vector>

namespace rank_predictor {
  constexpr int NUM_PLAYERS = 4;
  // 点数分布
  using Dist = std::valarray<int>;

  std::vector<std::vector<double>> predict_rank(const Dist& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                double ryukyoku,
                                                int max_steps,
                                                int oya);
}

#endif
