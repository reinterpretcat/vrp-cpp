#pragma once

#include "models/solution/Registry.hpp"
#include "models/solution/Route.hpp"
#include "models/solution/Tour.hpp"

#include <memory>
#include <range/v3/all.hpp>

namespace vrp::models::solution {

/// Creates a deep copy of shared route.
struct deep_copy_route final {
  std::shared_ptr<Route> operator()(const std::shared_ptr<const Route>& route) const {
    return std::make_shared<Route>(Route{route->actor, route->tour.copy()});
  }
};

/// Creates a deep copy of shared registry.
struct deep_copy_registry final {
  std::shared_ptr<Registry> operator()(const std::shared_ptr<const Registry>& registry) const {
    return std::make_shared<Registry>(Registry{*registry});
  }
};
}