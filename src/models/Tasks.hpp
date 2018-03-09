#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
struct Tasks {

  /// Customer id of the task.
  thrust::device_vector<int> ids;

  /// Cost of performing task.
  thrust::device_vector<float> costs;

  /// Current vehicle. Negative is a marker of unprocessed.
  thrust::device_vector<int> vehicles;

  /// Current vehicle capacity.
  thrust::device_vector<int> capacities;

  /// Current vehicle traveling time.
  thrust::device_vector<int> times;

  /// Reserves tasks size.
  void reserve(std::size_t populationSize, std::size_t problemSize) {
    auto size = populationSize * problemSize;
    ids.reserve(size);
    costs.reserve(size);
    vehicles.reserve(size);
    capacities.reserve(size);
    times.reserve(size);
  }
};

}
}

#endif //VRP_TASKS_HPP
