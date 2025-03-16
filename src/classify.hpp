#ifndef CLASSIFY_HPP
#define CLASSIFY_HPP

#include <valarray>

namespace rank_predictor::detail {
  int classfiy(const std::valarray<int>& scores);
}

#endif
