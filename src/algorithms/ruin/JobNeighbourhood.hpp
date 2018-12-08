#pragma once

#include "models/common/Timestamp.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Actor.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::ruin {

/// Calculates job neighborhood.
struct JobNeighbourhood final {
  explicit JobNeighbourhood(const models::problem::Fleet& fleet, const models::costs::TransportCosts& transport) {}

  ranges::view_all<models::problem::Job> neighbors(const models::solution::Actor& actor,
                                                   const models::problem::Job& job,
                                                   const models::common::Timestamp time) const {
    // TODO
  }
};
}