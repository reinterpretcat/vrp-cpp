#ifndef VRP_MODELS_RESOURCES_HPP
#define VRP_MODELS_RESOURCES_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represents resources to solve VRP.
struct Resources {
  /// Maximum vehicle capacity (units).
  thrust::device_vector<int> capacities;

  /// Vehicle cost per distance.
  thrust::device_vector<float> distanceCosts;

  /// Vehicle cost per traveling time.
  thrust::device_vector<float> timeCosts;

  /// Vehicle cost per waiting time.
  thrust::device_vector<float> waitingCosts;

  /// Vehicle time limit.
  thrust::device_vector<int> timeLimits;

  /// Reserves resource size.
  void reserve(std::size_t size) {
    capacities.reserve(size);
    distanceCosts.reserve(size);
    timeCosts.reserve(size);
    waitingCosts.reserve(size);
    timeLimits.reserve(size);
  }
};

}
}

#endif //VRP_MODELS_RESOURCES_HPP
