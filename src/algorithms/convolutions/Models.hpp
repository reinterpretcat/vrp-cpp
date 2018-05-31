#ifndef VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP

#include "models/Convolution.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Represents shared settings to work with convolutions algorithms.
struct Settings final {
  /// Specifies median ratio.
  float MedianRatio;
  /// Specifies ratio which controls threshold for grouping tasks.
  float ConvolutionRatio;
  /// Object pool
  vrp::utils::Pool& pool;
};

/// Represents a convolution joint pair.
struct JointPair final {
  /// Amount of shared customers.
  int similarity;
  /// Amount of unique customers served by convolution pair.
  int completeness;
  /// A pair constructed from two different convolutions.
  thrust::pair<vrp::models::Convolution, vrp::models::Convolution> pair;
};

/// Represents collection of join pairs with some meta information.
struct JointPairs final {
  /// Dimensions of the sets.
  thrust::pair<size_t, size_t> dimens;
  /// Represent convolution joint pair collection retrieved from pool.
  std::unique_ptr<thrust::device_vector<JointPair>, vrp::utils::Pool::Deleter> pairs;
};

/// Contains model shadows.
struct Model final {
  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
