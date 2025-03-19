#ifndef RANK_PREDICTOR_HPP
#define RANK_PREDICTOR_HPP

#include <valarray>
#include <vector>

namespace rank_predictor {
  std::vector<std::vector<double>> predict_rank(const std::valarray<int>& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                double ryukyoku,
                                                int max_steps,
                                                int oya);
}

#endif
