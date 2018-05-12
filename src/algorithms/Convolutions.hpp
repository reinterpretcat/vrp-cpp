#ifndef VRP_ALGORITHMS_CONVOLUTIONS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_HPP

#include "models/Convolution.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "utils/Pool.hpp"

#include <memory>

namespace vrp {
namespace algorithms {

/// Creates a group of customers (convolution) which can be served virtually as one.
struct create_best_convolutions final {

  /// Represent convolution collection retrieved  most likely from pool.
  using Convolutions = std::unique_ptr<thrust::device_vector<vrp::models::Convolution>,
                                       vrp::utils::Pool::Deleter>;

  /// Represents settings to create best convolutions from solution.
  struct Settings final {
    /// Specifies median ratio.
    float MedianRatio;
    /// Specifies ratio which controls threshold for grouping tasks.
    float ConvolutionRatio;
    /// Solution in tasks to be processed.
    int solution;
  };

  Convolutions operator()(const vrp::models::Problem &problem,
                          vrp::models::Tasks &tasks,
                          Settings &settings,
                          vrp::utils::Pool &pool);
};

}
}

#endif //VRP_ALGORITHMS_CONVOLUTIONS_HPP