#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "iterators/CartesianProduct.hpp"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

using namespace vrp::models;
using namespace vrp::algorithms::convolutions;

namespace {

/// Swaps iterator values.
template<typename Iterator>
__device__ inline void swap_values(Iterator a, Iterator b) {
  auto tmp = *a;
  *a = *b;
  *b = tmp;
}

/// Reverses iterator values.
template<typename Iterator>
__device__ void reverse_values(Iterator first, Iterator last) {
  for (; first != last && first != --last; ++first)
    swap_values(first, last);
}

/// Implements permutation generator (code is mostly taken from STL).
template<typename BidirectionalIterator, typename Compare>
__device__ bool next_permutation(BidirectionalIterator first,
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

/// Calculates rank of the pair.
struct rank_pair final {
  __device__ inline int operator()(const JointPair& pair) const {
    return pair.completeness - pair.similarity;
  }
};

/// Copies pair to convolution list.
struct copy_pair final {
  thrust::device_ptr<Convolution> convolutions;
  int current;

  __device__ inline int operator()(const JointPair& pair) {
    *(convolutions + current++) = pair.pair.first;
    *(convolutions + current++) = pair.pair.second;
    return 0;
  }
};

/// Calculates rank of the sliced pairs.
struct rank_sliced_pairs final {
  thrust::pair<size_t, size_t> dimens;

  template<typename KeyIterator, typename ValueIterator, typename RankOp>
  __device__ int operator()(const KeyIterator keyBegin,
                            const KeyIterator keyEnd,
                            const ValueIterator valueBegin,
                            RankOp rankOp) {
    int row = 0, rank = 0;
    for (auto it = keyBegin; it != keyEnd; ++it, ++row) {
      auto column = *it;
      if (row >= dimens.first) {
        rank += rankOp(*(valueBegin + (column % dimens.second)));
        continue;
      }

      if (column >= dimens.second) {
        rank += rankOp(*(valueBegin + row * dimens.second));
        continue;
      }

      auto index = row * dimens.second + column;
      rank += rankOp(*(valueBegin + index));
    }
    return rank;
  }
};

/// Finds and sets the best pairs slice to convolution container.
__global__ void setBestSlice(const thrust::device_ptr<JointPair> pairs,
                             const thrust::device_ptr<int> keys,
                             const thrust::pair<size_t, size_t> dimens,
                             thrust::device_ptr<Convolution> convolutions) {
  auto size = thrust::max(dimens.first, dimens.second);
  auto keyBegin = keys;
  auto keyEnd = keys + size;

  int maxRank = -1;
  do {
    auto rank = rank_sliced_pairs{dimens}(keyBegin, keyEnd, pairs, rank_pair{});
    if (rank > maxRank) {
      maxRank = rank;
      rank_sliced_pairs{dimens}(keyBegin, keyEnd, pairs, copy_pair{convolutions, 0});
    }
  } while (next_permutation(keyBegin, keyEnd, thrust::less<int>()));
}

}  // namespace

Convolutions create_sliced_convolutions::operator()(const Problem& problem,
                                                    Tasks& tasks,
                                                    const Settings& settings,
                                                    const JointPairs& pairs) const {
  auto size = thrust::max(pairs.dimens.first, pairs.dimens.second);
  auto keys =
    settings.pool.acquire<thrust::device_vector<int>>(static_cast<size_t>(tasks.customers));
  thrust::sequence(thrust::device, keys->begin(), keys->begin() + size, 0, 1);

  auto convolutionSize = thrust::min(pairs.dimens.first, pairs.dimens.second) * 2 +
                         std::abs(pairs.dimens.first - pairs.dimens.second);
  auto convolutions = settings.pool.acquire<thrust::device_vector<Convolution>>(convolutionSize);

  setBestSlice<<<1, 1>>>(pairs.pairs->data(), keys->data(), pairs.dimens, convolutions->data());

  return std::move(convolutions);
}
