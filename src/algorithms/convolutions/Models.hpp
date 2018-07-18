#ifndef VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP

#include "models/Convolution.hpp"
#include "runtime/UniquePointer.hpp"

namespace vrp {
namespace algorithms {
namespace convolutions {

/// Represents shared settings to work with convolutions algorithms.
struct Settings final {
  /// Specifies median ratio.
  float MedianRatio;
  /// Specifies ratio which controls threshold for grouping tasks.
  float ConvolutionRatio;
};

/// Specifies convolutions collection acquired from pool with its real size.
struct Convolutions {
  /// Convolutions size.
  size_t size;
  /// Convolutions data.
  vrp::runtime::unique_ptr<vrp::runtime::vector_ptr<vrp::models::Convolution>> data;
};

/// Represents collection of join pairs with some meta information.
struct JointPairs final {
  /// Dimensions of the sets.
  thrust::pair<size_t, size_t> dimens;
  /// Represent convolution joint pair collection retrieved from pool.
  vrp::runtime::unique_ptr<vrp::runtime::vector_ptr<vrp::models::JointPair>> data;
};

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
