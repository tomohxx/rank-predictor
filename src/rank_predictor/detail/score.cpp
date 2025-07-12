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

  // 100点未満を切り上げて100で割る
  int ceil(const int n)
  {
    return (n + 99) / 100;
  }

  // 1000点未満を四捨五入する
  int round(const int n)
  {
    return (n + 5) / 10;
  }

  // 1000点未満を四捨五入する
  std::pair<int, int> round(const std::pair<int, int>& p)
  {
    return {round(p.first), round(p.second)};
  }

  int round_if(const int n, const bool b)
  {
    return b ? round(n) : n * 100;
  }

  std::pair<int, int> round_if(const std::pair<int, int>& p, const bool b)
  {
    return b ? round(p) : std::make_pair(p.first * 100, p.second * 100);
  }
}

namespace rank_predictor::detail {
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

  std::pair<int, int> calc_score_tsumo(const int fu, const int han, const bool is_oya, const bool rounding)
  {
    return round_if(calc_score_tsumo(fu, han, is_oya), rounding);
  }

  int calc_score_ron(int fu, int han, bool is_oya, bool rounding)
  {
    return round_if(calc_score_ron(fu, han, is_oya), rounding);
  }
}
