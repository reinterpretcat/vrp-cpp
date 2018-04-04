#ifndef VRP_MODELS_TRANSITION_HPP
#define VRP_MODELS_TRANSITION_HPP

#include <thrust/execution_policy.h>

namespace vrp {
namespace models {

/// Represents transition from task to a customer.
struct Transition {

  /// Details about transition.
  struct Details {
    /// Task from which transition should happen.
    int from;
    /// Task to which transition should happen.
    int to;
    /// Customer which is being served by transition
    int customer;
    /// Vehicle used in transition.
    int vehicle;
  };

  /// Delta change of transition.
  struct Delta {
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
    __host__ __device__
    int duration() const {
      return traveling + serving + waiting;
    }
  };

  /// Details about transition.
  Details details;
  /// Delta change of transition.
  Delta delta;

  __host__ __device__
  Transition() : Transition({ -1, -1} , {-1, -1, -1, -1, -1 }) {}

  __host__ __device__
  Transition(const Details &details, const Delta &delta) :
    details(details), delta(delta) {}

  __host__ __device__
  Transition(const Transition&) = default;

  __host__ __device__
  Transition& operator=(const Transition&) = default;

  /// Flag whether transition is valid.
  __host__ __device__
  bool isValid() const {
    return details.customer > 0;
  }
};

/// Stores transition and its cost.
using TransitionCost = thrust::tuple<vrp::models::Transition, float>;

}
}

#endif //VRP_MODELS_TRANSITION_HPP
