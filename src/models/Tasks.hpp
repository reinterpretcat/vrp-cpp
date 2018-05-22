#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
/// VRP solition is represented as collection of tasks.
struct Tasks final {
  /// Stores device pointers to data.
  struct Shadow final {
    int customers;
    thrust::device_ptr<int> ids;
    thrust::device_ptr<float> costs;
    thrust::device_ptr<int> times;
    thrust::device_ptr<int> capacities;
    thrust::device_ptr<int> vehicles;
    thrust::device_ptr<bool> plan;
  };

  explicit Tasks() = default;

  explicit Tasks(int customers) : Tasks(customers, customers) {}

  explicit Tasks(int customers, int taskSize) :
    customers(customers), ids(), costs(), times(), capacities(), vehicles(), plan() {
    resize(static_cast<std::size_t>(taskSize));
  }

  /// Customers amount.
  int customers = 0;

  /// Customer id of the task.
  thrust::device_vector<int> ids;

  /// Cost of performing task.
  thrust::device_vector<float> costs;

  /// Departure time after serving task.
  thrust::device_vector<int> times;

  /// Remaining demand capacity.
  thrust::device_vector<int> capacities;

  /// Current vehicle. Negative is a marker of unprocessed.
  thrust::device_vector<int> vehicles;

  /// Keeps state of customer's state.
  thrust::device_vector<bool> plan;

  /// Returns size of tasks.
  int size() const { return static_cast<int>(ids.size()); }

  /// Returns size of population.
  int population() const { return size() / customers; }

  /// Resizes tasks size.
  void resize(std::size_t size) {
    ids.resize(size, -1);
    costs.resize(size, -1);
    times.resize(size, -1);
    capacities.resize(size, -1);
    vehicles.resize(size, -1);
    plan.resize(size, false);
  }

  /// Returns shadow object.
  Shadow getShadow() {
    return {customers,         ids.data(),      costs.data(), times.data(),
            capacities.data(), vehicles.data(), plan.data()};
  }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_TASKS_HPP
