#ifndef VRP_MODELS_RESOURCES_HPP
#define VRP_MODELS_RESOURCES_HPP

#include "runtime/Config.hpp"

namespace vrp {
namespace models {

/// Represents resources to solve VRP.
struct Resources final {
  /// Stores device pointers to data.
  struct Shadow final {
    int vehicles;
    vrp::runtime::vector_const_ptr<int> capacities;
    vrp::runtime::vector_const_ptr<float> distanceCosts;
    vrp::runtime::vector_const_ptr<float> timeCosts;
    vrp::runtime::vector_const_ptr<float> waitingCosts;
    vrp::runtime::vector_const_ptr<float> fixedCosts;
    vrp::runtime::vector_const_ptr<int> timeLimits;
  };

  /// Maximum vehicle capacity (units).
  vrp::runtime::vector<int> capacities;

  /// Vehicle cost per distance.
  vrp::runtime::vector<float> distanceCosts;

  /// Vehicle cost per traveling time.
  vrp::runtime::vector<float> timeCosts;

  /// Vehicle cost per waiting time.
  vrp::runtime::vector<float> waitingCosts;

  /// Vehicle fixed cost.
  vrp::runtime::vector<float> fixedCosts;

  /// TODO do we need it? Depot time window seems simulate it.
  /// Vehicle time limit.
  vrp::runtime::vector<int> timeLimits;

  /// Reserves resource size.
  void reserve(std::size_t size) {
    capacities.reserve(size);
    distanceCosts.reserve(size);
    timeCosts.reserve(size);
    waitingCosts.reserve(size);
    fixedCosts.reserve(size);
    timeLimits.reserve(size);
  }

  /// Returns shadow object.
  Shadow getShadow() const {
    return {static_cast<int>(capacities.size()),
            capacities.data(),
            distanceCosts.data(),
            timeCosts.data(),
            waitingCosts.data(),
            fixedCosts.data(),
            timeLimits.data()};
  }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_RESOURCES_HPP
