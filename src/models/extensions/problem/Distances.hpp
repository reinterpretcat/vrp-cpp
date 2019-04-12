#pragma once

#include "models/costs/TransportCosts.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Job.hpp"
#include "models/problem/Vehicle.hpp"

#include <range/v3/all.hpp>

namespace vrp::models::problem {

/// Calculates smallest possible distance between two jobs.
struct job_distance final {
  const costs::TransportCosts& transport;
  const common::Profile profile;
  const common::Timestamp departure;

  /// Returns min distance between location and job.
  common::Distance operator()(const Job& job, common::Location location) const {
    // NOTE ignore direction
    return ranges::min(getLocations(job) | ranges::view::transform([&](const auto& l) {
                         return l.has_value() ? transport.distance(profile, location, l.value(), departure) : 0;
                       }));
  }

  /// Returns min distance between two jobs.
  common::Distance operator()(const Job& lhs, const Job& rhs) const {
    using namespace ranges;

    auto left = getLocations(lhs) | to_vector;
    auto right = getLocations(rhs) | to_vector;

    return ranges::min(view::concat(view::cartesian_product(left, right) | view::transform([&](const auto& tuple) {
                                      auto l = std::get<0>(tuple);
                                      auto r = std::get<1>(tuple);
                                      return l.has_value() && r.has_value()
                                        ? transport.distance(profile, l.value(), r.value(), departure)
                                        : 0;
                                    })));
  }

private:
  using LocationView = ranges::any_view<std::optional<common::Location>>;

  LocationView getLocations(const Job& job) const {
    return analyze_job<LocationView>(
      job,
      [](const std::shared_ptr<const Service>& service) -> LocationView {
        return ranges::view::for_each(service->details, [](const auto& d) { return ranges::yield(d.location); });
      },
      [](const std::shared_ptr<const Sequence>& sequence) -> LocationView {
        return ranges::view::for_each(sequence->services, [](const auto& service) {
          return ranges::view::for_each(service->details, [](const auto& d) { return ranges::yield(d.location); });
        });
      });
  }
};
}