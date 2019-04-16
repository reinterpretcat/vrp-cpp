#pragma once

#include "models/Solution.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Fleet.hpp"

#include <memory>
#include <range/v3/all.hpp>

namespace vrp::test {

/// Finds vehicle from fleet by its id.
struct find_vehicle_by_id final {
  std::shared_ptr<const models::problem::Vehicle> operator()(const models::problem::Fleet& fleet,
                                                             const std::string& id) const {
    return (fleet.vehicles() | ranges::view::remove_if([&](const auto& v) {
              return std::any_cast<std::string>(v->dimens.find("id")->second) != id;
            }) |
            ranges::to_vector)
      .front();
  }
};

/// Finds route from solution  with specific vehicle.
struct find_route_by_vehicle_id {
  std::optional<std::shared_ptr<const models::solution::Route>> operator()(const models::Solution& solution,
                                                                           const std::string& id) const {
    auto routes = solution.routes | ranges::view::filter([&id](const auto& r) {
                    return models::problem::get_vehicle_id{}(*r->actor->vehicle) == id;
                  }) |
      ranges::to_vector;
    assert(routes.size() < 2);

    return routes.empty() ? std::make_shared<const models::solution::Route>() : routes.front();
  }
};
}