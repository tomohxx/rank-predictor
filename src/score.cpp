#include "score.hpp"
#include <cassert>
constexpr int MANGAN = 5;
constexpr int HANEMAN = 7;
constexpr int BAIMAN = 10;
constexpr int SANBAIMAN = 12;

namespace {
  std::pair<int, int> less_than_2000_tsumo(int base_score, bool is_oya);
  std::pair<int, int> greater_than_2000_tsumo(int han, bool is_oya);
  int less_than_2000_ron(int base_score, bool is_oya);
  int greater_than_2000_ron(int han, bool is_oya);

  inline int calc_base_score(const int fu, const int han)
  {
    return fu * (1 << (han + 2));
  }

  // 100点未満を切り上げる
  inline int ceil(const int n)
  {
    return (n + 99) / 100;
  }

  // 1000点未満を四捨五入する
  inline int round(const int n)
  {
    return (n + 5) / 10;
  }
}

namespace rank_predictor::detail {
  std::pair<int, int> calc_score_tsumo(int fu, int han, bool is_oya)
  {
    assert(fu != 0 && han != 0);

    const int base_score = calc_base_score(fu, han);

    if (base_score < 2000) {
      return less_than_2000_tsumo(base_score, is_oya);
    }
    else {
      return greater_than_2000_tsumo(han, is_oya);
    }
  }

  int calc_score_ron(int fu, int han, bool is_oya)
  {
    assert(fu != 0 && han != 0);

    const int base_score = calc_base_score(fu, han);

    if (base_score < 2000) {
      return less_than_2000_ron(base_score, is_oya);
    }
    else {
      return greater_than_2000_ron(han, is_oya);
    }
  }
}

namespace {
  // NOTE: 1000点未満の点数を切り捨て, 点数を1000で割った値を返す

  std::pair<int, int> less_than_2000_tsumo(const int base_score, const bool is_oya)
  {
    if (is_oya) {
      return std::make_pair(round(ceil(base_score * 2)), 0);
    }
    else {
      return std::make_pair(round(ceil(base_score)), round(ceil(base_score * 2)));
    }
  }

  int less_than_2000_ron(const int base_score, const bool is_oya)
  {
    if (is_oya) {
      return round(ceil(base_score * 6));
    }
    else {
      return round(ceil(base_score * 4));
    }
  }

  std::pair<int, int> greater_than_2000_tsumo(const int han, const bool is_oya)
  {
    if (han <= MANGAN) {
      return is_oya ? std::make_pair(4, 0) : std::make_pair(2, 4);
    }
    else if (han <= HANEMAN) {
      return is_oya ? std::make_pair(6, 0) : std::make_pair(3, 6);
    }
    else if (han <= BAIMAN) {
      return is_oya ? std::make_pair(8, 0) : std::make_pair(4, 8);
    }
    else if (han <= SANBAIMAN) {
      return is_oya ? std::make_pair(12, 0) : std::make_pair(6, 12);
    }
    else {
      return is_oya ? std::make_pair(16, 0) : std::make_pair(8, 16);
    }
  }

  int greater_than_2000_ron(const int han, const bool is_oya)
  {
    if (han <= MANGAN) {
      return is_oya ? 12 : 8;
    }
    else if (han <= HANEMAN) {
      return is_oya ? 18 : 12;
    }
    else if (han <= BAIMAN) {
      return is_oya ? 24 : 16;
    }
    else if (han <= SANBAIMAN) {
      return is_oya ? 36 : 24;
    }
    else {
      return is_oya ? 48 : 32;
    }
  }
}
