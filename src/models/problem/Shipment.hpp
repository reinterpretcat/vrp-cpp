#pragma once

#include "models/common/Dimension.hpp"
#include "models/common/Duration.hpp"
#include "models/common/Location.hpp"
#include "models/common/TimeWindow.hpp"

#include <vector>

namespace vrp::models::problem {

/// Represents a job associated with two locations.
struct Shipment final {
  /// Represents a work which has to be performed.
  struct Service final {
    /// Location of the service.
    vrp::models::common::Location location;

    /// Time has to be spend performing work.
    vrp::models::common::Duration duration;

    /// Time windows when work can be performed.
    std::vector<vrp::models::common::TimeWindow> times;
  };

  /// Specifies shipment id.
  std::string id;

  /// Pickup service performed before delivery.
  Service pickup;

  /// Delivery service performed after pickup.
  Service delivery;

  /// Dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimensions;
};

}  // namespace vrp::models::problem
