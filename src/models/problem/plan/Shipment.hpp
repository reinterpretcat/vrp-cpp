#pragma once

#include "models/problem/plan/Service.hpp"
#include "models/problem/plan/Job.hpp"

#include <optional>

namespace vrp::models::problem::plan {

/// Represents a job associated with two locations.
struct Shipment final : public Job {
  /// Pickup service performed before delivery.
  Service pickup;

  /// Delivery service performed after pickup.
  Service delivery;

  void accept(JobVisitor &visitor) const override {
    visitor.visit(*this);
  }
};

}
