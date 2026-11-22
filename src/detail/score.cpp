#include "score.hpp"
#include <algorithm>
#include <cassert>
constexpr int MANGAN = 5;
constexpr int HANEMAN = 7;
constexpr int BAIMAN = 10;
constexpr int SANBAIMAN = 12;

namespace {
  int calc_base_score(const int fu, const int han)
  {
    if (han <= MANGAN) {
      return std::min(fu * (1 << (han + 2)), 2000);
    }
    else if (han <= HANEMAN) {
      return 3000;
    }
    else if (han <= BAIMAN) {
      return 4000;
    }
    else if (han <= SANBAIMAN) {
      return 6000;
    }
    else {
      return 8000;
    }
  }

  // 100点未満を切り上げる
  int ceil(const int n)
  {
    return (n + 99) / 100 * 100;
  }
}

namespace mahjong::rank_predictor::detail {
  std::pair<int, int> calc_score_tsumo(const int fu, const int han, const bool is_oya)
  {
    assert(fu != 0 && han != 0);

    const int base_score = calc_base_score(fu, han);

    if (is_oya) {
      return std::make_pair(ceil(base_score * 2), 0);
    }
    else {
      return std::make_pair(ceil(base_score), ceil(base_score * 2));
    }
  }

  int calc_score_ron(const int fu, const int han, const bool is_oya)
  {
    assert(fu != 0 && han != 0);

    const int base_score = calc_base_score(fu, han);

    if (is_oya) {
      return ceil(base_score * 6);
    }
    else {
      return ceil(base_score * 4);
    }
  }
}
