#pragma once

#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Job.hpp"

#include <range/v3/all.hpp>

namespace vrp::models::problem {

/// Calculates smallest possible distance between two jobs.
struct job_distance final {
  const costs::TransportCosts& transport;
  const std::string& profile;
  const common::Timestamp departure;

  common::Distance operator()(const Job& lhs, const Job& rhs) const {
    using namespace ranges;
    auto left = getLocations(lhs) | to_vector;
    auto right = getLocations(rhs) | to_vector;

    auto distance =
      ranges::min(view::concat(view::cartesian_product(left, right) | view::transform([&](const auto& tuple) {
                                 return transport.distance(profile, std::get<0>(tuple), std::get<1>(tuple), departure);
                               }),
                               view::single(common::NoDistance)));

    return distance < common::NoDistance ? distance : 0;
  }

private:
  ranges::any_view<common::Location> getLocations(const Job& job) const {
    return analyze_job<ranges::any_view<common::Location>>(
      job,
      [](const std::shared_ptr<const Service>& service) -> ranges::any_view<common::Location> {
        return ranges::view::for_each(service->details, [](const auto& d) {
          return ranges::yield_if(d.location.has_value(), d.location.has_value() ? d.location.value() : 0);
        });
      },
      [](const std::shared_ptr<const Sequence>& sequence) -> ranges::any_view<common::Location> {
        throw std::domain_error("not implemented");
      });
  }
};
}