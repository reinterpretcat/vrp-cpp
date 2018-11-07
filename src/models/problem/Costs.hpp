#pragma once

namespace vrp::models::problem {

/// Represents costs for actor.
struct Costs final {
  /// A fixed cost to use an actor.
  double fixed;

  /// Cost per distance unit.
  double perDistance;

  /// Cost per driving time unit.
  double perDrivingTime;

  /// Cost per waiting time unit.
  double perWaitingTime;

  /// Cost per service time unit.
  double perServiceTime;
};

}  // namespace vrp::models::problem::fleet
