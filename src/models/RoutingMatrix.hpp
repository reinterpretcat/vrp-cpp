#ifndef VRP_MODELS_ROUTINGMATRIX_HPP
#define VRP_MODELS_ROUTINGMATRIX_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Contains routing matrix data.
struct RoutingMatrix final {
  /// Stores device pointers to data.
  struct Shadow {
    thrust::device_ptr<const float> distances;
    thrust::device_ptr<const int> durations;
  };

  /// Matrix of distances.
  thrust::device_vector<float> distances;

  /// Matrix of durations.
  thrust::device_vector<int> durations;

  /// Reserves resource size.
  void reserve(std::size_t size) {
    distances.reserve(size);
    durations.reserve(size);
  }

  /// Returns shadow object.
  Shadow getShadow() const { return {distances.data(), durations.data()}; }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_ROUTINGMATRIX_HPP
