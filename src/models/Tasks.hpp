#ifndef VRP_MODEL_TASKS_HPP
#define VRP_MODEL_TASKS_HPP

#include "models/Plan.hpp"
#include "runtime/Config.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace models {

/// Represent task by "Struct of Array" idiom.
/// VRP solition is represented as collection of tasks.
struct Tasks final {
  /// Stores device pointers to data.
  struct Shadow final {
    int customers;
    vrp::runtime::vector_ptr<int> ids;
    vrp::runtime::vector_ptr<float> costs;
    vrp::runtime::vector_ptr<int> times;
    vrp::runtime::vector_ptr<int> capacities;
    vrp::runtime::vector_ptr<int> vehicles;
    vrp::runtime::vector_ptr<Plan> plan;
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
  vrp::runtime::vector<int> ids;

  /// Cost of performing task.
  vrp::runtime::vector<float> costs;

  /// Departure time after serving task.
  vrp::runtime::vector<int> times;

  /// Remaining demand capacity.
  vrp::runtime::vector<int> capacities;

  /// Current vehicle. Negative is a marker of unprocessed.
  vrp::runtime::vector<int> vehicles;

  /// Keeps state of customer's state.
  vrp::runtime::vector<Plan> plan;

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
    plan.resize(size, Plan::empty());
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
