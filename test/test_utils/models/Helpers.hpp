#pragma once

#include "models/problem/Fleet.hpp"

#include <memory>

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
}