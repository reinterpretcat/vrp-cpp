#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "iterators/CartesianProduct.hpp"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

using namespace vrp::models;
using namespace vrp::runtime;
using namespace vrp::algorithms::convolutions;

namespace {

/// Swaps iterator values.
template<typename Iterator>
EXEC_UNIT void swap_values(Iterator a, Iterator b) {
  // TODO why I cannot simply use Iterator::value_type or auto?
  vector<int>::value_type tmp = *a;

  *a = *b;
  *b = tmp;
}

/// Reverses iterator values.
template<typename Iterator>
EXEC_UNIT void reverse_values(Iterator first, Iterator last) {
  for (; first != last && first != --last; ++first)
    swap_values(first, last);
}

/// Implements permutation generator (code is mostly taken from STL).
template<typename BidirectionalIterator, typename Compare>
EXEC_UNIT bool next_permutation(BidirectionalIterator first,
                                BidirectionalIterator last,
                                Compare comp) {
  if (first == last) return false;
  BidirectionalIterator next = first;
  ++next;
  if (next == last) return false;
  next = last;
  --next;

  for (;;) {
    BidirectionalIterator ii = next;
    --next;
    if (comp(*next, *ii)) {
      BidirectionalIterator mid = last;
      while (!comp(*next, *--mid)) {}
      swap_values(next, mid);
      reverse_values(ii, last);
      return true;
    }
    if (next == first) {
      reverse_values(first, last);
      return false;
    }
  }
}

template <typename RandomIterator>
EXEC_UNIT void random_shuffle(RandomIterator first, RandomIterator last) {
  typename std::iterator_traits<RandomIterator>::difference_type i, n;
  n = last - first;

  // TODO pass generator outside
  auto rand = thrust::minstd_rand();
  auto dist = thrust::random::uniform_int_distribution<int>(0, static_cast<int>(n - 1));

  for (i = n-1; i > 0; --i) {
    RandomIterator a = first + i;
    RandomIterator b = first + dist(rand) % (i+1);
    swap_values(a, b);
  }
}

/// Calculates rank of the pair.
struct rank_pair final {
  EXEC_UNIT inline int operator()(const JointPair& pair, const thrust::pair<bool, bool>&) const {
    return pair.completeness - pair.similarity;
  }
};

/// Copies pair to convolution list.
struct copy_pair final {
  vector_ptr<Convolution>& convolutions;
  int current;

  EXEC_UNIT inline int operator()(const JointPair& pair, const thrust::pair<bool, bool>& copyMap) {
    if (copyMap.first) *(convolutions + current++) = pair.pair.first;
    if (copyMap.second) *(convolutions + current++) = pair.pair.second;
    return 0;
  }
};

/// Calculates rank of the sliced pairs.
struct rank_sliced_pairs final {
  thrust::pair<size_t, size_t> dimens;

  template<typename KeyIterator, typename ValueIterator, typename RankOp>
  EXEC_UNIT int operator()(const KeyIterator keyBegin,
                           const KeyIterator keyEnd,
                           const ValueIterator valueBegin,
                           RankOp rankOp) {
    int row = 0, rank = 0;
    for (auto it = keyBegin; it != keyEnd; ++it, ++row) {
      auto column = *it;
      if (row >= dimens.first) {
        rank += rankOp(*(valueBegin + (column % dimens.second)), thrust::make_pair(false, true));
        continue;
      }

      if (column >= dimens.second) {
        rank += rankOp(*(valueBegin + row * dimens.second), thrust::make_pair(true, false));
        continue;
      }

      auto index = row * dimens.second + column;
      rank += rankOp(*(valueBegin + index), thrust::make_pair(true, true));
    }
    return rank;
  }
};

/// Scans all possible permutations and finds the best.
EXEC_UNIT void findBest(const vector_ptr<JointPair> pairs,
                       const vector_ptr<int> keyBegin,
                       const vector_ptr<int> keyEnd,
                       const thrust::pair<size_t, size_t> dimens,
                       vector_ptr<Convolution> convolutions) {
  int maxRank = -1;
  do {
    auto rank = rank_sliced_pairs{dimens}(keyBegin, keyEnd, pairs, rank_pair{});
    if (rank > maxRank) {
      maxRank = rank;
      rank_sliced_pairs{dimens}(keyBegin, keyEnd, pairs, copy_pair{convolutions, 0});
    }
  } while (next_permutation(keyBegin, keyEnd, thrust::less<int>()));
}

/// Shuffles keys in random order.
EXEC_UNIT void iAmLucky(const vector_ptr<JointPair> pairs,
                        const vector_ptr<int> keyBegin,
                        const vector_ptr<int> keyEnd,
                        const thrust::pair<size_t, size_t> dimens,
                        vector_ptr<Convolution> convolutions) {

  random_shuffle(keyBegin, keyEnd);
  rank_sliced_pairs{dimens}(keyBegin, keyEnd, pairs, copy_pair{convolutions, 0});
}

/// Finds and sets the best pairs slice to convolution container.
EXEC_UNIT void setBestSlice(const vector_ptr<JointPair> pairs,
                            const vector_ptr<int> keys,
                            const thrust::pair<size_t, size_t> dimens,
                            vector_ptr<Convolution> convolutions) {
  auto size = thrust::max(dimens.first, dimens.second);
  auto keyBegin = keys;
  auto keyEnd = keys + size;

  // NOTE we have size! permutations, so limit its usage
  if (size > 7) iAmLucky(pairs, keyBegin, keyEnd, dimens, convolutions);
  else findBest(pairs, keyBegin, keyEnd, dimens, convolutions);
}

}  // namespace

EXEC_UNIT Convolutions create_sliced_convolutions::operator()(const Settings& settings,
                                                              const JointPairs& pairs) const {
  auto size = thrust::max(pairs.dimens.first, pairs.dimens.second);
  auto keys = make_unique_ptr_data<int>(static_cast<size_t>(solution.problem.size));
  auto convolutions = make_unique_ptr_data<Convolution>(pairs.dimens.first + pairs.dimens.second);

  thrust::sequence(exec_unit_policy{}, *keys, *keys + size, 0, 1);

  setBestSlice(*pairs.data, *keys, pairs.dimens, *convolutions);

  return {pairs.dimens.first + pairs.dimens.second, std::move(convolutions)};
}
