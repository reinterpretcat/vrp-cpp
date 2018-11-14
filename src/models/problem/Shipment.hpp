#pragma once

#include "models/common/Dimension.hpp"
#include "models/problem/Detail.hpp"

#include <vector>

namespace vrp::models::problem {

/// Represents a job which consists of two dependent parts: pickup and delivery.
struct Shipment final {
  /// Specifies shipment id.
  std::string id;

  /// Pickup service performed before delivery.
  Detail pickup;

  /// Delivery service performed after pickup.
  Detail delivery;

  /// Dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimensions;
};

}  // namespace vrp::models::problem
