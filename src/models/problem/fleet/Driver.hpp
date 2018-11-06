#pragma once

namespace vrp::models::problem::fleet {

/// Represents a driver, person who drives Vehicle
/// Introduced to allow future extensions:
/// * reuse vehicle multiple times with different drivers
/// * solve best driver-vehicle match problem.
struct Driver {
  /// Specifies driver costs.
  Costs costs;

  /// Specifies departure/arrival schedule limits.
  vrp::models::common::Schedule schedule;
};

}  // namespace vrp::models::problem::fleet
