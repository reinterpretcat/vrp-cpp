#include "algorithms/convolutions/JointConvolutions.hpp"
#include "iterators/CartesianProduct.cu"
#include "utils/Memory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/discard_iterator.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::models;
using namespace vrp::iterators;
using namespace vrp::utils;
using namespace thrust::placeholders;

namespace {
/// Joins two convolutions into pair.
struct create_joint_pair final {
  Model model;

  /// Counts equal pairs of ints.
  struct count_equal_pairs final {
    int* total;
    __device__ int operator()(int l, int r) {
      if (l == r) atomicAdd(total, 1);
      return 0;
    }
  };

  __host__ __device__
  JointPair operator()(const Convolution &left, const Convolution &right) const {
    // NOTE task range is inclusive, so +1 is required
    auto leftSize = static_cast<size_t>(left.tasks.second - left.tasks.first + 1);
    auto rightSize = static_cast<size_t>(right.tasks.second - right.tasks.first + 1);

    auto leftBegin = model.tasks.ids + left.base + left.tasks.first;
    auto rightBegin = model.tasks.ids + right.base + right.tasks.first;

    typedef thrust::device_ptr<int> Iterator;
    vrp::iterators::repeated_range<Iterator> repeated(leftBegin, leftBegin + leftSize, rightSize);
    vrp::iterators::tiled_range<Iterator> tiled(rightBegin, rightBegin + rightSize, rightSize);

    int *total = (int*) malloc(sizeof(int)); *total = 0;

    thrust::transform(
        thrust::device,
        repeated.begin(), repeated.end(),
        tiled.begin(),
        thrust::make_discard_iterator(),
        count_equal_pairs { total }
    );

    auto rank = *total; free(total);
    auto served = leftSize + rightSize - rank;
    auto completeness = static_cast<float>(served) / model.tasks.customers;

    return JointPair {rank, completeness, {left, right}};
  }
};
}

JointPairs create_joint_convolutions::operator()(const Problem &problem, Tasks &tasks,
                                                 const Settings &settings,
                                                 const Convolutions &left, const Convolutions &right) const {
  typedef thrust::device_vector<Convolution>::const_iterator Iterator;
  repeated_range<Iterator> repeated(left->begin(), left->end(), right->size());
  tiled_range<Iterator> tiled(right->begin(), right->end(), right->size());

  // theoretical max convolution size in each group
  auto size = static_cast<int>(1 / settings.ConvolutionRatio);
  auto pairs = settings.pool.acquire<thrust::device_vector<JointPair>>(static_cast<size_t>(size * size));

  // create all possible combinations from two group
  thrust::transform(
      thrust::device,
      repeated.begin(), repeated.end(),
      tiled.begin(),
      pairs->begin(),
      create_joint_pair { {problem.getShadow(), tasks.getShadow() } }
  );

  return pairs;
}
