#ifndef VRP_MODELS_ROUTINGMATRIX_HPP
#define VRP_MODELS_ROUTINGMATRIX_HPP

#include "runtime/Config.hpp"

namespace vrp {
namespace models {

/// Contains routing matrix data.
struct RoutingMatrix final {
  /// Stores device pointers to data.
  struct Shadow {
    vrp::runtime::vector_const_ptr<float> distances;
    vrp::runtime::vector_const_ptr<int> durations;
  };

  /// Matrix of distances.
  vrp::runtime::vector<float> distances;

  /// Matrix of durations.
  vrp::runtime::vector<int> durations;

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
