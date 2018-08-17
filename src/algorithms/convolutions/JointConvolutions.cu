#include "algorithms/convolutions/JointConvolutions.hpp"
#include "iterators/CartesianProduct.hpp"

#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::iterators;
using namespace vrp::runtime;
using namespace thrust::placeholders;

namespace {
/// Joins two convolutions into pair.
struct create_joint_pair final {
  Solution::Shadow solution;

  /// Counts equal pairs of ints.
  struct count_equal_pairs final {
    int* total;
    EXEC_UNIT int operator()(int l, int r) {
      if (l == r) vrp::runtime::add(total, 1);
      return 0;
    }
  };

  EXEC_UNIT JointPair operator()(const Convolution& left, const Convolution& right) const {
    // NOTE task range is inclusive, so +1 is required
    auto leftSize = static_cast<size_t>(left.tasks.second - left.tasks.first + 1);
    auto rightSize = static_cast<size_t>(right.tasks.second - right.tasks.first + 1);

    auto leftBegin = solution.tasks.ids + left.base + left.tasks.first;
    auto rightBegin = solution.tasks.ids + right.base + right.tasks.first;

    typedef vector_ptr<int> Iterator;
    vrp::iterators::repeated_range<Iterator> repeated(leftBegin, leftBegin + leftSize, rightSize);
    vrp::iterators::tiled_range<Iterator> tiled(rightBegin, rightBegin + rightSize, rightSize);

    int* total = (int*) malloc(sizeof(int));
    *total = 0;

    thrust::transform(exec_unit_policy{}, repeated.begin(), repeated.end(), tiled.begin(),
                      thrust::make_discard_iterator(), count_equal_pairs{total});

    auto shared = *total;
    free(total);
    auto served = static_cast<int>(leftSize + rightSize - shared);

    return JointPair{shared, served, {left, right}};
  }
};
}  // namespace

EXEC_UNIT JointPairs create_joint_convolutions::operator()(const Settings& settings,
                                                           const Convolutions& left,
                                                           const Convolutions& right) const {
  auto leftData = *left.data;
  auto leftSize = left.size;
  auto rightData = *right.data;
  auto rightSize = right.size;

  repeated_range<decltype(leftData)> repeated(leftData, leftData + leftSize, rightSize);
  tiled_range<decltype(leftData)> tiled(rightData, rightData + rightSize, rightSize);

  // theoretical max convolution size in each group
  auto size = solution.tasks.customers / settings.ConvolutionSize;
  auto pairs = make_unique_ptr_data<JointPair>(static_cast<size_t>(size * size));

  // create all possible combinations from two group
  thrust::transform(exec_unit_policy{}, repeated.begin(), repeated.end(), tiled.begin(), *pairs,
                    create_joint_pair{solution});

  return {{leftSize, rightSize}, std::move(pairs)};
}
