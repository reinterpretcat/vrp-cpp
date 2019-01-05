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
    return std::make_shared<Route>(Route{
      route->actor,
      copy(*route->start),
      copy(*route->end),
      copy(route->tour),
    });
  }

private:
  Tour::Activity copy(const Activity& activity) const { return std::make_shared<Activity>(Activity{activity}); }

  Tour copy(const Tour& tour) const {
    auto newTour = Tour{};
    ranges::for_each(tour.activities(), [&](const auto& activity) { newTour.add(copy(*activity)); });
    return std::move(newTour);
  }
};

/// Creates a deep copy of shared registry.
struct deep_copy_registry final {
  std::shared_ptr<Registry> operator()(const std::shared_ptr<const Registry>& registry) const {
    return std::make_shared<Registry>(Registry{*registry});
  }
};
}