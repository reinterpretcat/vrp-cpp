#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "iterators/CartesianProduct.hpp"

#include <thrust/execution_policy.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

using namespace vrp::models;
using namespace vrp::algorithms::convolutions;

namespace {

/// Swaps iterator values.
template<typename Iterator>
__host__ __device__ inline void swap_iter_values(Iterator a, Iterator b) {
  auto tmp = *a;
  *a = *b;
  *b = tmp;
}

/// Implements permutation generator (code is mostly taken from STL).
template<typename BidirectionalIterator, typename Compare>
__host__ __device__ bool next_permutation(BidirectionalIterator first,
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
      swap_iter_values(next, mid);
      // TODO device policy
      thrust::reverse(ii, last);
      return true;
    }
    if (next == first) {
      // TODO device policy
      thrust::reverse(first, last);
      return false;
    }
  }
}

struct Pair {
  int l;
  int r;
};

struct create_pairs final {
  __host__ __device__ Pair operator()(int l, int r) { return {l, r}; }
};

thrust::host_vector<Pair> makePairs(thrust::host_vector<int>& left,
                                    thrust::host_vector<int>& right) {
  typedef typename thrust::host_vector<int>::iterator Iterator;
  vrp::iterators::repeated_range<Iterator> repeated(left.begin(), left.end(), right.size());
  vrp::iterators::tiled_range<Iterator> tiled(right.begin(), right.end(), right.size());

  thrust::host_vector<Pair> pairs(right.size() * left.size());

  thrust::transform(thrust::host, repeated.begin(), repeated.end(), tiled.begin(), pairs.begin(),
                    create_pairs{});

  return pairs;
}

struct rank_slice_pairs final {
  size_t rows;
  size_t columns;

  template<typename KeyIterator, typename ValueIterator>
  __host__ __device__ int operator()(const KeyIterator keyBegin,
                                     const KeyIterator keyEnd,
                                     const ValueIterator valueBegin) {
    int row = 0;
    for (auto it = keyBegin; it != keyEnd; ++it, ++row) {
      auto column = *it;
      if (row >= rows) {
        auto value = *(valueBegin + (column % columns));
        printf("(%d)", value.r);
        continue;
      }

      if (column >= columns) {
        auto value = *(valueBegin + row * columns);
        printf("(%d)", value.l);
        continue;
      }

      auto index = row * columns + column;
      auto value = *(valueBegin + index);
      printf("(%d,%d)", value.l, value.r);
    }
    printf("\n");
    return 0;
  }
};

void generatePermutations(const thrust::host_vector<Pair>& pairs,
                          size_t rows, size_t columns) {
  thrust::host_vector<int> keys(thrust::max(rows, columns));
  thrust::sequence(keys.begin(), keys.end(), 0, 1);

  // expected:
  // (1,4) (1,5) (1,6)   (2,4) (2,5) (2,6)   (3,4) (3,5) (3,6)   (x,4) (x,5) (x,6)
  std::cout << "pairs:\n";
   // thrust::for_each(pairs.begin(), pairs.end(), [](const Pair &pair) {
   //   std::cout << "(" << pair.l << "," << pair.r << ")";
   // });
  for(auto i = 0; i < pairs.size(); ++i) {
    auto& pair = pairs[i];
    if (i % columns == 0) {
      std::cout << "\t";
    }
    std::cout << "(" << pair.l << "," << pair.r << ")";
  }
  std::cout << "\n";

  // iterate through all permutations to get a best slice
  int maxRank = -1;
  //Convolutions convolutions(size);
  do {
    std::cout << "=>";
    thrust::copy(keys.begin(), keys.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    auto rank = rank_slice_pairs{rows,columns}(keys.begin(), keys.end(), pairs.begin());
    //if (rank > maxRank) {
    //  // TODO copy slice
    //}

  } while (next_permutation(keys.begin(), keys.end(), thrust::less<int>()));

  //return convolutions;
}

}  // namespace

Convolutions create_sliced_convolutions::operator()(/*const Problem& problem,
                                                    Tasks& tasks,
                                                    const Settings& settings,
                                                    const JointPairs&*/) const {
  std::vector<int> vecLeft = {1, 2, 3, 4};
  std::vector<int> vecRight = {5, 6};

//  std::vector<int> vecLeft = {1, 2, 3};
//  std::vector<int> vecRight = {4, 5, 6, 7};

  thrust::host_vector<int> left(vecLeft.begin(), vecLeft.end());
  thrust::host_vector<int> right(vecRight.begin(), vecRight.end());

  auto pairs = makePairs(left, right);

  generatePermutations(pairs, left.size(), right.size());

  // TODO
  return Convolutions();
}
