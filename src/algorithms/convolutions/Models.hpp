#ifndef VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
#define VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP

#include "models/Convolution.hpp"
#include "utils/memory/DevicePool.hpp"
#include "utils/memory/DeviceUnique.hpp"

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
  using ConvolutionsPtr = thrust::device_ptr<vrp::models::Convolution>;
  using Deleter = vrp::utils::DevicePool::TypedPool<vrp::models::Convolution>::pool_vector_deleter;

  /// Convolutions size
  size_t size;
  /// Convolutions data.
  vrp::utils::device_unique_ptr<ConvolutionsPtr, Deleter> data;
};

/// Represents collection of join pairs with some meta information.
struct JointPairs final {
  using JointPairPtr = thrust::device_ptr<vrp::models::JointPair>;
  using Deleter = vrp::utils::DevicePool::TypedPool<vrp::models::JointPair>::pool_vector_deleter;

  /// Dimensions of the sets.
  thrust::pair<size_t, size_t> dimens;
  /// Represent convolution joint pair collection retrieved from pool.
  vrp::utils::device_unique_ptr<JointPairPtr, Deleter> data;
};

///// Convolution candidates.
// using JointConvolutions = thrust::pair<Convolutions, Convolutions>;

}  // namespace convolutions
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_ALGORITHMS_CONVOLUTIONS_MODELS_HPP
