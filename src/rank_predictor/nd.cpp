#include "nd.hpp"
#include "common.hpp"
#include "detail/classify.hpp"
#include <algorithm>
#include <array>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>
namespace ublas = boost::numeric::ublas;

namespace {
  // 3 x 3 正定値対称行列のコレスキー分解
  ublas::matrix<double> cholesky3(const ublas::matrix<double>& A)
  {
    ublas::matrix<double> L(3, 3, 0.);

    for (std::size_t i = 0u; i < 3u; ++i) {
      for (std::size_t j = 0u; j <= i; ++j) {
        double s = A(i, j);

        for (std::size_t k = 0u; k < j; ++k) {
          s -= L(i, k) * L(j, k);
        }

        L(i, j) = (i == j ? std::sqrt(s) : s / L(j, j));
      }
    }

    return L;
  }

  std::tuple<ublas::vector<double>, ublas::matrix<double>> make_gap_dist(const ublas::vector<double>& mu,
                                                                         const ublas::matrix<double>& covariance,
                                                                         const std::array<int, 4>& order)
  {
    // 点数の分布を指定された順位の隣接点差の分布に変換する行列
    ublas::matrix<double> A(3, 4, 0.);

    for (std::size_t i = 0u; i < 3; ++i) {
      A(i, order[i]) = 1.;
      A(i, order[i + 1u]) = -1.;
    }

    // 点差の平均
    ublas::vector<double> gap_mu = ublas::prod(A, mu);
    ublas::matrix<double> AC = ublas::prod(A, covariance);
    // 点差の分散共分散行列
    ublas::matrix<double> gap_covariance = ublas::prod(AC, ublas::trans(A));

    return {std::move(gap_mu), std::move(gap_covariance)};
  }

  double integrate_dist(const ublas::vector<double>& gap_mu,
                        const ublas::matrix<double>& gap_covariance)
  {
    // 隣接点差の3成分がすべて正となる確率（指定順位順になる確率）を求める.
    // gap_covariance = L * L^T とコレスキー分解し, 点差を gap_mu + L * [x, y, z]^T と表す.
    // x, y, z は独立な標準正規変数.

    const auto L = cholesky3(gap_covariance);
    const boost::math::normal_distribution<double> normal;

    using gk = boost::math::quadrature::gauss_kronrod<double, 31>;

    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double tolerance = 1e-8;
    constexpr unsigned max_depth = 12;

    const double lower_x = -gap_mu(0) / L(0, 0);

    return gk::integrate([&](const double x) {
      const double lower_y = -(gap_mu(1) + L(1, 0) * x) / L(1, 1);

      return boost::math::pdf(normal, x) * gk::integrate([&](const double y) {
               const double lower_z = -(gap_mu(2) + L(2, 0) * x + L(2, 1) * y) / L(2, 2);

               return boost::math::pdf(normal, y) * boost::math::cdf(boost::math::complement(normal, lower_z));
             },
                                                         lower_y, inf, max_depth, tolerance);
    },
                         lower_x, inf, max_depth, tolerance);
  }
}

namespace rank_predictor::nd {
  std::tuple<ublas::vector<double>, ublas::matrix<double>> propagate(const ublas::vector<double>& mu,
                                                                     const ublas::matrix<double>& covariance,
                                                                     const Patterns& patterns,
                                                                     const double ryukyoku,
                                                                     const int oya)
  {
    // 点数移動の平均
    ublas::vector<double> delta_mu(4, 0.);
    // 点数移動の 2 次モーメント
    ublas::matrix<double> second_moment(4, 4, 0.);

    for (const auto& pattern : patterns) {
      const auto delta = calc_delta(pattern.actor, pattern.target, oya, pattern.fu, pattern.han, false);
      ublas::vector<double> tmp(4);

      std::copy(std::cbegin(delta), std::cend(delta), tmp.begin());

      delta_mu += tmp;
      second_moment += ublas::outer_prod(tmp, tmp);
    }

    delta_mu *= (1. - ryukyoku) / static_cast<double>(patterns.size());
    second_moment *= (1. - ryukyoku) / static_cast<double>(patterns.size());

    ublas::vector<double> next_mu = mu + delta_mu;
    ublas::matrix<double> next_covariance = covariance + second_moment - ublas::outer_prod(delta_mu, delta_mu);

    return {std::move(next_mu), std::move(next_covariance)};
  }

  std::vector<std::vector<double>> predict_rank(const std::valarray<int>& initial,
                                                const std::vector<int>& fu_list,
                                                const std::vector<int>& han_list,
                                                const double ryukyoku,
                                                const int max_steps,
                                                int oya)
  {
    const auto patterns = make_patterns(fu_list, han_list);

    ublas::vector<double> mu(4);
    ublas::matrix<double> covariance(4, 4, 0.);

    std::copy(std::cbegin(initial), std::cend(initial), mu.begin());

    for (int step = 0; step < max_steps; ++step) {
      std::tie(mu, covariance) = propagate(mu, covariance, patterns, ryukyoku, oya);
      // NOTE: 連荘は考慮しない
      oya = (oya + 1) % NUM_PLAYERS;
    }

    std::vector<std::vector<double>> probs(4, std::vector<double>(4, 0.));
    std::array<int, 4> order{0, 1, 2, 3};

    do {
      const auto [gap_mu, gap_covariance] = make_gap_dist(mu, covariance, order);
      const double prob = integrate_dist(gap_mu, gap_covariance);

      for (std::size_t rank = 0u; rank < 4u; ++rank) {
        // NOTE: order[rank]: rank + 1 位のプレイヤー id
        probs[order[rank]][rank] += prob;
      }
    } while (std::next_permutation(order.begin(), order.end()));

    return probs;
  }
}
