#include "dp.hpp"
#include "detail/classify.hpp"
#include "detail/score.hpp"
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <unordered_map>

namespace rank_predictor {
  constexpr int NUM_CLASSES = 24;
  constexpr int MAX_DELTA = 25;
  constexpr int MIN_DELTA = -25;
  constexpr int SCALE = 1000;

  // 原点
  const Dist origin(0., 4);

  struct Hash {
    std::size_t operator()(const Dist& dist) const
    {
      return boost::hash_range(std::cbegin(dist), std::cend(dist));
    }
  };

  struct Pred {
    bool operator()(const Dist& lhs, const Dist& rhs) const
    {
      return (lhs == rhs).min();
    }
  };

  // 点数分布が実現する確率
  using DistProb = std::unordered_map<Dist, double, Hash, Pred>;

  // 和了パターン
  struct Pattern {
    int actor;
    int target;
    int fu;
    int han;
  };

  using Patterns = std::vector<Pattern>;

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

  Dist calc_delta(const int actor, const int target, const int oya, const int fu, const int han)
  {
    Dist delta(0, NUM_PLAYERS);

    if (actor == target) {
      const auto [x, y] = detail::calc_score_tsumo(fu, han, actor == oya, true);

      for (int pid = 0; pid < NUM_PLAYERS; ++pid) {
        delta[pid] = (pid == actor ? (pid == oya ? x * 3 : x * 2 + y) : (pid == oya ? -y : -x));
      }
    }
    else {
      delta[actor] = detail::calc_score_ron(fu, han, actor == oya, true);
      delta[target] = -delta[actor];
    }

    return delta;
  }

  inline bool is_valid_dist(const Dist& dist)
  {
    return std::all_of(std::cbegin(dist), std::cend(dist), [](const auto& x) {
      return x >= MIN_DELTA && x < MAX_DELTA;
    });
  }

  Dist clip(const Dist& dist, const Dist& delta)
  {
    const Dist tmp = dist + delta;

    return is_valid_dist(tmp) ? tmp : dist;
  }

  DistProb propagate(const DistProb& current, const Patterns& patterns, const double ryukyoku, const int oya)
  {
    DistProb next;

    for (const auto& [source, prob] : current) {
      next[source] += ryukyoku * prob;

      for (const auto& pattern : patterns) {
        const auto delta = calc_delta(pattern.actor, pattern.target, oya, pattern.fu, pattern.han);
        const auto dest = clip(source, delta);

        next[dest] += (1. - ryukyoku) / patterns.size() * prob; // NOTE: 一様分布
      }
    }

    return next;
  }

  std::vector<double> classify(const Dist& initial, const DistProb& dist_prob)
  {
    std::vector<double> rank_classes(NUM_CLASSES, 0.);

    for (const auto& [dist, prob] : dist_prob) {
      const int rank_class = detail::classfiy(initial + dist * SCALE);

      rank_classes.at(rank_class) += prob;
    }

    return rank_classes;
  }

  std::vector<std::vector<double>> predict_rank(const Dist& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                const double ryukyoku,
                                                const int max_steps,
                                                int oya)
  {
    const auto patterns = make_patterns(fu_list, han_list);

    DistProb dist_prob;

    dist_prob[origin] = 1.;

    for (int step = 0; step < max_steps; ++step) {
      dist_prob = propagate(dist_prob, patterns, ryukyoku, oya);
      // NOTE: 連荘は考慮しない
      oya = (oya + 1) % NUM_PLAYERS;
    }

    const auto rank_classes = classify(initial, dist_prob);

    return {
        {
            rank_classes[0] + rank_classes[1] + rank_classes[2] + rank_classes[3] + rank_classes[4] + rank_classes[5],
            rank_classes[6] + rank_classes[7] + rank_classes[12] + rank_classes[13] + rank_classes[18] + rank_classes[19],
            rank_classes[8] + rank_classes[10] + rank_classes[14] + rank_classes[16] + rank_classes[20] + rank_classes[22],
            rank_classes[9] + rank_classes[11] + rank_classes[14] + rank_classes[15] + rank_classes[21] + rank_classes[23],
        },
        {
            rank_classes[6] + rank_classes[7] + rank_classes[8] + rank_classes[9] + rank_classes[10] + rank_classes[11],
            rank_classes[0] + rank_classes[1] + rank_classes[14] + rank_classes[15] + rank_classes[20] + rank_classes[21],
            rank_classes[2] + rank_classes[4] + rank_classes[12] + rank_classes[17] + rank_classes[18] + rank_classes[23],
            rank_classes[3] + rank_classes[5] + rank_classes[13] + rank_classes[16] + rank_classes[19] + rank_classes[22],
        },
        {
            rank_classes[12] + rank_classes[13] + rank_classes[14] + rank_classes[15] + rank_classes[16] + rank_classes[17],
            rank_classes[2] + rank_classes[3] + rank_classes[8] + rank_classes[9] + rank_classes[22] + rank_classes[23],
            rank_classes[0] + rank_classes[5] + rank_classes[6] + rank_classes[11] + rank_classes[19] + rank_classes[21],
            rank_classes[1] + rank_classes[4] + rank_classes[7] + rank_classes[10] + rank_classes[18] + rank_classes[20],
        },
        {
            rank_classes[18] + rank_classes[19] + rank_classes[20] + rank_classes[21] + rank_classes[22] + rank_classes[23],
            rank_classes[4] + rank_classes[5] + rank_classes[10] + rank_classes[11] + rank_classes[16] + rank_classes[17],
            rank_classes[1] + rank_classes[3] + rank_classes[7] + rank_classes[9] + rank_classes[13] + rank_classes[15],
            rank_classes[0] + rank_classes[2] + rank_classes[6] + rank_classes[8] + rank_classes[12] + rank_classes[14],
        },
    };
  }
}
