#ifndef VRP_MODELS_TRANSITION_HPP
#define VRP_MODELS_TRANSITION_HPP

#include <thrust/execution_policy.h>

namespace vrp {
namespace models {

/// Represents transition between customers.
struct Transition {
  /// Customer.
  int customer;
  /// Vehicle.
  int vehicle;

  /// Distance
  float distance;

  /// Traveling time.
  int traveling;
  /// Serving time.
  int serving;
  /// Waiting time.
  int waiting;
  /// Demand.
  int demand;

  /// Task from which transition is performed.
  int task;

  __host__ __device__
  Transition(int customer,
             int vehicle,
             float distance,
             int traveling,
             int serving,
             int waiting,
             int demand,
             int task) :
      customer(customer), vehicle(vehicle),  distance(distance),
      traveling(traveling), serving(serving), waiting(waiting),
      demand(demand), task(task) { }

  __host__ __device__
  Transition() : Transition(-1, -1, -1, -1, -1, -1, -1, -1) {}

  __host__ __device__
  Transition(const Transition&) = default;

  __host__ __device__
  Transition& operator=(const Transition&) = default;

  /// Returns duration of transition.
  __host__ __device__
  int duration() const {
    return traveling + serving + waiting;
  }

  /// Flag whether transition is valid.
  __host__ __device__
  bool isValid() const {
    return task > 0;
  }

  /// Creates invalid transition.
  __host__ __device__
  static Transition createInvalid() {
    return Transition();
  }
};

}
}

#endif //VRP_MODELS_TRANSITION_HPP
