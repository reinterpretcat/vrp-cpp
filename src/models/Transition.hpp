#ifndef VRP_MODELS_TRANSITION_HPP
#define VRP_MODELS_TRANSITION_HPP

#include "models/Convolution.hpp"
#include "runtime/Config.hpp"
#include "runtime/DeviceVariant.hpp"

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
    vrp::runtime::device_variant<int, Convolution> customer;
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
    ANY_EXEC_UNIT int duration() const { return traveling + serving + waiting; }
  };

  /// Details about transition.
  Details details;
  /// Delta change of transition.
  Delta delta;

  ANY_EXEC_UNIT Transition() : Transition({-1, -1}, {-1, -1, -1, -1, -1}) {}

  ANY_EXEC_UNIT Transition(const Details& details, const Delta& delta) :
    details(details), delta(delta) {}

  ANY_EXEC_UNIT Transition(const Transition&) = default;

  ANY_EXEC_UNIT Transition& operator=(const Transition&) = default;

  /// Flag whether transition is valid.
  ANY_EXEC_UNIT bool isValid() const { return details.from >= 0; }
};

}  // namespace models
}  // namespace vrp

#endif  // VRP_MODELS_TRANSITION_HPP
