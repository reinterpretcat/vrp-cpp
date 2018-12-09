#pragma once

#include "models/common/Timestamp.hpp"
#include "models/costs/TransportCosts.hpp"
#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Actor.hpp"

#include <map>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

namespace vrp::algorithms::ruin {

/// Calculates job neighborhood in terms of the cost.
struct JobNeighbourhood final {
  explicit JobNeighbourhood(const models::problem::Fleet& fleet, const models::costs::TransportCosts& transport) {}

  ranges::any_view<models::problem::Job> neighbors(const models::solution::Actor& actor,
                                                   const models::problem::Job& job,
                                                   const models::common::Timestamp time) const {
    // TODO
  }

private:
  auto getProfiles(const models::problem::Fleet& fleet) {
    return fleet.vehicles() | ranges::view::transform([](const auto& v) { return v->id; }) |  //
      ranges::to_vector | ranges::action::sort | ranges::action::unique;
  }

  std::map<std::string, std::vector<models::problem::Job>> index_;
};
}