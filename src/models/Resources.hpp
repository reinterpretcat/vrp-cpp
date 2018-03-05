#ifndef VRP_MODELS_RESOURCES_HPP
#define VRP_MODELS_RESOURCES_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represents resources to solve VRP.
struct Resources {

  /// Vehicle quantities per type.
  thrust::device_vector<int> vehicleAvalabilities;


  /// Maximum vehicle capacity (units).
  thrust::device_vector<int> vehicleCapacities;

  /// Vehicle cost per distance.
  thrust::device_vector<float> vehicleDistanceCosts;

  /// Vehicle cost per traveling time.
  thrust::device_vector<float> vehicleTimeCosts;

  /// Vehicle cost per waiting time.
  thrust::device_vector<float> vehicleWaitingCosts;

  /// Vehicle time limit.
  thrust::device_vector<int> vehicleTimeLimits;
};

}
}

#endif //VRP_MODELS_RESOURCES_HPP
