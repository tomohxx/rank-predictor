#include "classify.hpp"
#include <bit>
#include <cassert>

namespace {
  inline unsigned int compare(const int i, const int j, const unsigned int b)
  {
    return i >= j ? b : ~b;
  }

  inline unsigned int query01(const int score0, const int score1)
  {
    return compare(score0, score1, 0b010011'010011'000000'111111u);
  }

  inline unsigned int query02(const int score0, const int score2)
  {
    return compare(score0, score2, 0b000111'000000'010011'111111u);
  }

  inline unsigned int query03(const int score0, const int score3)
  {
    return compare(score0, score3, 0b000000'000111'000111'111111u);
  }

  inline unsigned int query12(const int score1, const int score2)
  {
    return compare(score1, score2, 0b001101'000000'111111'010011u);
  }

  inline unsigned int query13(const int score1, const int score3)
  {
    return compare(score1, score3, 0b000000'001101'111111'000111u);
  }

  inline unsigned int query23(const int score2, const int score3)
  {
    return compare(score2, score3, 0b000000'111111'001101'001101u);
  }
}

namespace rank_predictor::detail {
  int classfiy(const std::valarray<int>& scores)
  {
    assert(scores.size() == 4u);

    const unsigned int b = query01(scores[0], scores[1]) &
                           query02(scores[0], scores[2]) &
                           query03(scores[0], scores[3]) &
                           query12(scores[1], scores[2]) &
                           query13(scores[1], scores[3]) &
                           query23(scores[2], scores[3]);

    assert(std::popcount(b & 0x00FFFFFFu) == 1);

    return std::countr_zero(b);
  }
}
