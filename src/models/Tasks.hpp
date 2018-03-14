#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
struct Tasks final {
  /// Stores device pointers to data.
  struct Shadow final {
    thrust::device_ptr<int> ids;
    thrust::device_ptr<float> costs;
    thrust::device_ptr<int> times;
    thrust::device_ptr<int> vehicles;
  };

  explicit Tasks() = default;

  explicit Tasks(int size) {
    resize(static_cast<std::size_t>(size));
  }

  /// Customer id of the task.
  thrust::device_vector<int> ids;

  /// Cost of performing task.
  thrust::device_vector<float> costs;

  /// Total time for performing task..
  thrust::device_vector<int> times;

  /// Current vehicle. Negative is a marker of unprocessed.
  thrust::device_vector<int> vehicles;

  /// Returns size of tasks.
  int size() const {
    return static_cast<int>(ids.size());
  }

  /// Resizes tasks size.
  void resize(std::size_t size) {
    ids.resize(size, -1);
    costs.resize(size, -1);
    vehicles.resize(size, -1);
    times.resize(size, -1);
  }

  /// Returns shadow object.
  Shadow getShadow() {
    return {ids.data(),
            costs.data(),
            times.data(),
            vehicles.data()};
  }
};

}
}

#endif //VRP_TASKS_HPP
