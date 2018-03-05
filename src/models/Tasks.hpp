#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
struct Tasks {

  /// Customer id of the task.
  thrust::device_vector<int> customerIds;


  /// Cost of performing task.
  thrust::device_vector<float> costs;


  /// Vehicle sequential id.
  thrust::device_vector<int> vehicleIds;

  /// Current vehicle capacity.
  thrust::device_vector<int> vehicleCapacities;

  /// Current vehicle traveling time.
  thrust::device_vector<int> vehicleTimes;

  /// Current vehicle type.
  thrust::device_vector<int> vehicleTypes;
};

}
}

#endif //VRP_TASKS_HPP
