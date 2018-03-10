#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
struct Tasks {

  explicit Tasks() = default;

  explicit Tasks(int size) {
    auto s = static_cast<unsigned long>(size);
    ids.reserve(s);
    costs.reserve(s);
    vehicles.reserve(s);
    capacities.reserve(s);
    times.reserve(s);
  }

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

  /// Returns size of tasks.
  int size() const {
    return static_cast<int>(ids.size());
  }
};

}
}

#endif //VRP_TASKS_HPP
