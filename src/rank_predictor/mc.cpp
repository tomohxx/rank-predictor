#include "mc.hpp"
#include "common.hpp"
#include "detail/classify.hpp"

namespace rank_predictor {
  Dist propagate(const Dist& source,
                 const Patterns& patterns,
                 const double ryukyoku,
                 const int oya,
                 std::mt19937_64& engine)
  {
    if (std::uniform_real_distribution<> dist(0., 1.); dist(engine) < ryukyoku) {
      return source;
    }

    std::uniform_int_distribution<> dist(0, patterns.size() - 1u);

    const std::size_t pid = dist(engine);
    const auto& pattern = patterns[pid];
    const auto delta = calc_delta(pattern.actor, pattern.target, oya, pattern.fu, pattern.han, false);
    const auto dest = source + delta;

    return dest;
  }

  std::vector<std::vector<double>> predict_rank(const std::valarray<int>& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                const double ryukyoku,
                                                const int max_steps,
                                                int oya,
                                                const int num_playout,
                                                std::mt19937_64& engine)
  {
    const auto patterns = make_patterns(fu_list, han_list);

    std::vector<int> rank_classes(NUM_CLASSES, 0);

    for (int i = 0; i < num_playout; ++i) {
      Dist dist = initial;

      for (int step = 0; step < max_steps; ++step) {
        dist = propagate(dist, patterns, ryukyoku, oya, engine);
        // NOTE: 連荘は考慮しない
        oya = (oya + 1) % NUM_PLAYERS;
      }

      const int rank_class = detail::classfiy(dist);

      ++rank_classes.at(rank_class);
    }

    return {
        {
            static_cast<double>(rank_classes[0] + rank_classes[1] + rank_classes[2] + rank_classes[3] + rank_classes[4] + rank_classes[5]) / num_playout,
            static_cast<double>(rank_classes[6] + rank_classes[7] + rank_classes[12] + rank_classes[13] + rank_classes[18] + rank_classes[19]) / num_playout,
            static_cast<double>(rank_classes[8] + rank_classes[10] + rank_classes[14] + rank_classes[16] + rank_classes[20] + rank_classes[22]) / num_playout,
            static_cast<double>(rank_classes[9] + rank_classes[11] + rank_classes[14] + rank_classes[15] + rank_classes[21] + rank_classes[23]) / num_playout,
        },
        {
            static_cast<double>(rank_classes[6] + rank_classes[7] + rank_classes[8] + rank_classes[9] + rank_classes[10] + rank_classes[11]) / num_playout,
            static_cast<double>(rank_classes[0] + rank_classes[1] + rank_classes[14] + rank_classes[15] + rank_classes[20] + rank_classes[21]) / num_playout,
            static_cast<double>(rank_classes[2] + rank_classes[4] + rank_classes[12] + rank_classes[17] + rank_classes[18] + rank_classes[23]) / num_playout,
            static_cast<double>(rank_classes[3] + rank_classes[5] + rank_classes[13] + rank_classes[16] + rank_classes[19] + rank_classes[22]) / num_playout,
        },
        {
            static_cast<double>(rank_classes[12] + rank_classes[13] + rank_classes[14] + rank_classes[15] + rank_classes[16] + rank_classes[17]) / num_playout,
            static_cast<double>(rank_classes[2] + rank_classes[3] + rank_classes[8] + rank_classes[9] + rank_classes[22] + rank_classes[23]) / num_playout,
            static_cast<double>(rank_classes[0] + rank_classes[5] + rank_classes[6] + rank_classes[11] + rank_classes[19] + rank_classes[21]) / num_playout,
            static_cast<double>(rank_classes[1] + rank_classes[4] + rank_classes[7] + rank_classes[10] + rank_classes[18] + rank_classes[20]) / num_playout,
        },
        {
            static_cast<double>(rank_classes[18] + rank_classes[19] + rank_classes[20] + rank_classes[21] + rank_classes[22] + rank_classes[23]) / num_playout,
            static_cast<double>(rank_classes[4] + rank_classes[5] + rank_classes[10] + rank_classes[11] + rank_classes[16] + rank_classes[17]) / num_playout,
            static_cast<double>(rank_classes[1] + rank_classes[3] + rank_classes[7] + rank_classes[9] + rank_classes[13] + rank_classes[15]) / num_playout,
            static_cast<double>(rank_classes[0] + rank_classes[2] + rank_classes[6] + rank_classes[8] + rank_classes[12] + rank_classes[14]) / num_playout,
        },
    };
  }
}
