#pragma once

#include "models/problem/Service.hpp"

#include <vector>

namespace vrp::models::problem {

/// Represents a job which consists of multiple sub jobs without ids.
/// All of these jobs must be performed in the order specified or none of them.
struct Shipment final {
  /// A list of sub jobs which must be performed in order specified.
  std::vector<Service> jobs;

  /// Common shipment dimensions which simulates work requirements.
  vrp::models::common::Dimensions dimens;
};

}  // namespace vrp::models::problem
