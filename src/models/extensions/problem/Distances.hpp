#pragma once

#include "models/costs/TransportCosts.hpp"
#include "models/problem/Job.hpp"
#include "utils/extensions/Variant.hpp"

#include <range/v3/all.hpp>

namespace vrp::models::problem {

/// Calculates smallest possible distance between two jobs.
struct job_distance final {
  const costs::TransportCosts& transport;
  const std::string& profile;
  const common::Timestamp departure;

  common::Distance operator()(const Job& lhs, const Job& rhs) const {
    using namespace ranges;

    static const auto fun = ranges::overload(
      [](const std::shared_ptr<const Service>& service) -> ranges::any_view<common::Location> {
        return view::for_each(service->details, [](const auto& d) {
          return ranges::yield(d.location.has_value() ? d.location.value() : 0);
        });
      },
      [](const std::shared_ptr<const Shipment>& shipment) -> ranges::any_view<common::Location> {
        throw std::domain_error("not implemented");
      });

    auto left = utils::mono_result(const_cast<problem::Job&>(lhs).visit(fun)) | to_vector;
    auto right = utils::mono_result(const_cast<problem::Job&>(rhs).visit(fun)) | to_vector;

    return ranges::min(view::cartesian_product(left, right) | view::transform([&](const auto& tuple) {
                         return transport.distance(profile, std::get<0>(tuple), std::get<1>(tuple), departure);
                       }));
  }
};
}