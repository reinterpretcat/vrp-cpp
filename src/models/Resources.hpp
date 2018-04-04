#ifndef VRP_MODELS_RESOURCES_HPP
#define VRP_MODELS_RESOURCES_HPP

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace models {

/// Represents resources to solve VRP.
struct Resources final {
  /// Stores device pointers to data.
  struct Shadow final {
    int vehicles;
    thrust::device_ptr<const int> capacities;
    thrust::device_ptr<const float> distanceCosts;
    thrust::device_ptr<const float> timeCosts;
    thrust::device_ptr<const float> waitingCosts;
    thrust::device_ptr<const float> fixedCosts;
    thrust::device_ptr<const int> timeLimits;
  };

  /// Maximum vehicle capacity (units).
  thrust::device_vector<int> capacities;

  /// Vehicle cost per distance.
  thrust::device_vector<float> distanceCosts;

  /// Vehicle cost per traveling time.
  thrust::device_vector<float> timeCosts;

  /// Vehicle cost per waiting time.
  thrust::device_vector<float> waitingCosts;

  /// Vehicle fixed cost.
  thrust::device_vector<float> fixedCosts;

  /// Vehicle time limit.
  thrust::device_vector<int> timeLimits;

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

}
}

#endif //VRP_MODELS_RESOURCES_HPP
