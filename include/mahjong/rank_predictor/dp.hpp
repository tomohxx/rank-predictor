#ifndef MAHJONG_RANK_PREDICTOR_DP_HPP
#define MAHJONG_RANK_PREDICTOR_DP_HPP

#include <valarray>
#include <vector>

namespace mahjong::rank_predictor::dp {
  std::vector<std::vector<double>> predict_rank(const std::valarray<int>& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                double ryukyoku,
                                                int max_steps,
                                                int oya);
}

#endif
