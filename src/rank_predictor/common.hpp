#include <valarray>
#include <vector>

namespace rank_predictor {
  constexpr int NUM_PLAYERS = 4;
  constexpr int NUM_CLASSES = 24;

  // 点数分布
  using Dist = std::valarray<int>;

  // 和了パターン
  struct Pattern {
    int actor;
    int target;
    int fu;
    int han;
  };

  using Patterns = std::vector<Pattern>;

  Patterns make_patterns(const std::vector<int>& fu_list, const std::vector<int>& han_list);
  Dist calc_delta(int actor, int target, int oya, int fu, int han, bool rounding);
}
