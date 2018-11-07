#pragma once

#include "Service.hpp"
#include "Job.hpp"

#include <optional>

namespace vrp::models::problem {

/// Represents a job associated with two locations.
struct Shipment final : public Job {

  /// Represents a work which has to be performed.
  struct Service final {
    /// Location of the service.
    vrp::models::common::Location location;

    /// Time has to be spend performing work.
    vrp::models::common::Duration duration;

    /// Time windows when work can be performed.
    std::vector<vrp::models::common::TimeWindow> times;
  };

  /// Pickup service performed before delivery.
  Service pickup;

  /// Delivery service performed after pickup.
  Service delivery;

  /// Dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimensions;

  void accept(JobVisitor &visitor) const override {
    visitor.visit(*this);
  }
};

}
