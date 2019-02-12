#pragma once

#include "algorithms/objectives/ObjectiveFunction.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::objectives {

/// Objective function which maximize job assignment by applying
/// penalty cost to each unassigned job.
template<int Penalty = 1000>
struct penalize_unassigned_jobs final : public ObjectiveFunction {
  /// Estimates solution returning total cost and included penalty.
  models::common::ObjectiveCost operator()(const models::Solution& sln,
                                           const models::costs::ActivityCosts& activity,
                                           const models::costs::TransportCosts& transport) const override {
    using namespace models::common;
    using namespace ranges;

    auto actual = ranges::accumulate(sln.routes, Cost{0}, [&](auto acc, const auto& r) {
      auto cost = acc + r->actor->vehicle->costs.fixed +
        activity.cost(*r->actor, *r->tour.start(), r->tour.start()->schedule.arrival);

      return ranges::accumulate(r->tour.activities() | view::sliding(2), cost, [&](auto inner, const auto& view) {
        auto [from, to] = std::tie(*std::begin(view), *(std::begin(view) + 1));
        return inner + activity.cost(*r->actor, *to, to->schedule.arrival) +
          transport.cost(*r->actor, from->detail.location, to->detail.location, from->schedule.departure);
      });
    });

    auto penalty = static_cast<Cost>(sln.unassigned.size() * Penalty);

    return {actual, penalty};
  }
};
}