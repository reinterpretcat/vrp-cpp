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

    return ranges::min(view::concat(view::cartesian_product(left, right) | view::transform([&](const auto& tuple) {
                                      auto lhs = std::get<0>(tuple);
                                      auto rhs = std::get<1>(tuple);
                                      return lhs.has_value() && rhs.has_value()
                                        ? transport.distance(profile, lhs.value(), rhs.value(), departure)
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