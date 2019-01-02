#pragma once

#include "models/Solution.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// Removes empty tours from solution.
struct remove_empty_tours final {
  void operator()(models::Solution& sln) const {
    ranges::action::remove_if(sln.routes, [&](const auto& route) {
      if (route->tour.empty()) {
        sln.registry->free(*route->actor);
        return true;
      }
      return false;
    });
  }
};
}
