#ifndef VRP_MODELS_TRANSITION_HPP
#define VRP_MODELS_TRANSITION_HPP

#include <thrust/execution_policy.h>

namespace vrp {
namespace models {

/// Represents transition between customers.
struct Transition {
  /// Customer.
  const int customer;
  /// Vehicle.
  const int vehicle;

  /// Distance
  const float distance;

  /// Traveling time.
  const int traveling;
  /// Serving time.
  const int serving;
  /// Waiting time.
  const int waiting;

  __host__ __device__
  Transition(int customer,
             int vehicle,
             float distance,
             int traveling,
             int serving,
             int waiting) :
      customer(customer), vehicle(vehicle), distance(distance),
      traveling(traveling), serving(serving), waiting(waiting) { }

  /// Returns duration of transition.
  __host__ __device__
  int duration() const {
    return traveling + serving + waiting;
  }

  /// Flag whether transition is valid.
  __host__ __device__
  bool isValid() const {
    return distance >= 0;
  }

  /// Creates invalid transition.
  __host__ __device__
  static Transition createInvalid() {
    return {-1, -1, -1 , -1, -1, -1};
  }

};

}
}

#endif //VRP_MODELS_TRANSITION_HPP
