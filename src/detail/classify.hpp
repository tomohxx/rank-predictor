#ifndef MAHJONG_RANK_PREDICTOR_DETAIL_CLASSIFY_HPP
#define MAHJONG_RANK_PREDICTOR_DETAIL_CLASSIFY_HPP

#include <valarray>

namespace mahjong::rank_predictor::detail {
  int classfiy(const std::valarray<int>& scores);
}

#endif
