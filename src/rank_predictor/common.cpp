#include "common.hpp"
#include "detail/score.hpp"

namespace rank_predictor {
  Patterns make_patterns(const std::vector<int>& fu_list, const std::vector<int>& han_list)
  {
    Patterns patterns;

    patterns.reserve(NUM_PLAYERS * NUM_PLAYERS * fu_list.size() * han_list.size());

    for (int i = 0; i < NUM_PLAYERS; ++i) {
      for (int j = 0; j < NUM_PLAYERS; ++j) {
        for (const int& fu : fu_list) {
          for (const int& han : han_list) {
            patterns.emplace_back(i, j, fu, han);
          }
        }
      }
    }

    return patterns;
  }

  Dist calc_delta(const int actor,
                  const int target,
                  const int oya,
                  const int fu,
                  const int han,
                  const bool rounding)
  {
    Dist delta(0, NUM_PLAYERS);

    if (actor == target) {
      const auto [x, y] = detail::calc_score_tsumo(fu, han, actor == oya, rounding);

      for (int pid = 0; pid < NUM_PLAYERS; ++pid) {
        delta[pid] = (pid == actor ? (pid == oya ? x * 3 : x * 2 + y) : (pid == oya ? -y : -x));
      }
    }
    else {
      delta[actor] = detail::calc_score_ron(fu, han, actor == oya, rounding);
      delta[target] = -delta[actor];
    }

    return delta;
  }
}
