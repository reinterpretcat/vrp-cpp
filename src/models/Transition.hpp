#ifndef VRP_MODELS_TRANSITION_HPP
#define VRP_MODELS_TRANSITION_HPP

#include "models/Convolution.hpp"
#include "utils/types/DeviceVariant.hpp"

#include <thrust/execution_policy.h>

namespace vrp {
namespace models {

/// Represents transition from task to a customer.
struct Transition {
  /// Stores information about state between transitions.
  struct State final {
    /// Last customer.
    int customer;
    /// Remaining demand.
    int capacity;
    /// Departure time.
    int time;
  };

  /// Details about transition.
  struct Details final {
    /// Base index.
    int base;
    /// Task from which transition should happen.
    int from;
    /// Task to which transition should happen.
    int to;
    /// Customer which is being served by transition
    /// represented by their id or convolution.
    vrp::utils::device_variant<int, Convolution> customer;
    /// Vehicle used in transition.
    int vehicle;
  };

  /// Delta change of transition.
  struct Delta final {
    /// Travelling distance.
    float distance;
    /// Traveling time.
    int traveling;
    /// Serving time.
    int serving;
    /// Waiting time.
    int waiting;
    /// Demand change.
    int demand;

    /// Returns total delta's duration.
    __host__ __device__ int duration() const { return traveling + serving + waiting; }
  };

  /// Details about transition.
  Details details;
  /// Delta change of transition.
  Delta delta;

  __host__ __device__ Transition() : Transition({-1, -1}, {-1, -1, -1, -1, -1}) {}

  __host__ __device__ Transition(const Details& details, const Delta& delta) :
    details(details), delta(delta) {}

  __host__ __device__ Transition(const Transition&) = default;

  __host__ __device__ Transition& operator=(const Transition&) = default;

  /// Flag whether transition is valid.
  __host__ __device__ bool isValid() const { return details.from >= 0; }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_TRANSITION_HPP
