#ifndef MAHJONG_RANK_PREDICTOR_DETAIL_SCORE_HPP
#define MAHJONG_RANK_PREDICTOR_DETAIL_SCORE_HPP

#include <utility>

namespace mahjong::rank_predictor::detail {
  std::pair<int, int> calc_score_tsumo(int fu, int han, bool is_oya);
  int calc_score_ron(int fu, int han, bool is_oya);
}

#endif
