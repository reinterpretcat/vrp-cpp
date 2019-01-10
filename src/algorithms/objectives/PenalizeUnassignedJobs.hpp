#pragma once

#include "algorithms/objectives/ObjectiveFunction.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::objectives {

/// Objective function which maximize job assignment by applying
/// penalty cost to each unassigned job.
template<int Penalty = 1000>
struct penalize_unassigned_jobs final : public ObjectiveFunction {
  /// Estimates solution returning total cost and included penalty.
  ObjectiveFunction::Result operator()(const models::Solution& sln,
                                       const models::costs::ActivityCosts& activity,
                                       const models::costs::TransportCosts& transport) const override {
    using namespace models::common;
    using namespace ranges;

    auto actual = ranges::accumulate(sln.routes, Cost{0}, [&](auto acc, const auto& r) {
      return ranges::accumulate(
        view::concat(view::single(r->start), r->tour.activities(), view::single(r->end)) | view::sliding(2),
        acc + r->actor->vehicle->costs.fixed + activity.cost(*r->actor, *r->start, r->start->schedule.arrival),
        [&](auto inner, const auto& view) {
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