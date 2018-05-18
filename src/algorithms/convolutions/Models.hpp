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

  // TODO to be excluded
  /// Solution in tasks to be processed.
  int solution;

  /// Object pool
  vrp::utils::Pool &pool;
};

/// Represents a convolution joint pair.
struct JointPair final {
  /// Amount of shared customers.
  int rank;
  /// Ratio of customers served by pair to all customers.
  float completeness;
  /// A pair constructed from two different convolutions.
  thrust::pair<vrp::models::Convolution, vrp::models::Convolution> pair;
};

/// Represent convolution joint pair collection retrieved from pool.
using JointPairs = std::unique_ptr<thrust::device_vector<JointPair>,
                                   vrp::utils::Pool::Deleter>;

/// Contains model shadows.
struct Model final {
  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
};

}
}
}

#endif //VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
