#ifndef MAHJONG_RANK_PREDICTOR_MC_HPP
#define MAHJONG_RANK_PREDICTOR_MC_HPP

#include <random>
#include <valarray>
#include <vector>

namespace mahjong::rank_predictor::mc {
  std::vector<std::vector<double>> predict_rank(const std::valarray<int>& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                double ryukyoku,
                                                int max_steps,
                                                int oya,
                                                const int num_playout,
                                                std::mt19937_64& engine);
}

#endif
