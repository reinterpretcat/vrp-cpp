#pragma once

#include "models/Solution.hpp"
#include "utils/Random.hpp"

#include <optional>

namespace vrp::models::solution {

/// Selects random job within its route from list of routes.
struct select_job final {
  using ReturnType = std::optional<std::pair<std::shared_ptr<solution::Route>, problem::Job>>;

  /// Returns no job if routes are empty or there is no single activity withing job.
  ReturnType operator()(const std::vector<std::shared_ptr<solution::Route>>& routes, utils::Random& random) const {
    if (routes.empty()) return ReturnType{};

    auto routeIndex = random.uniform<int>(0, static_cast<int>(routes.size()) - 1);

    auto ri = static_cast<size_t>(routeIndex);
    do {
      auto route = routes[routeIndex];

      if (!route->tour.empty()) {
        auto job = random_job(route, random);
        if (job) return ReturnType{std::make_pair(route, job.value())};
      }

      ri = (ri + 1) % routes.size();
    } while (ri != routeIndex);

    return ReturnType{};
  }

private:
  /// Selects random job from route.
  std::optional<problem::Job> random_job(const std::shared_ptr<solution::Route>& route, utils::Random& random) const {
    auto size = route->tour.sizes().second;
    if (size == 0) return {};

    auto activityIndex = random.uniform<int>(0, static_cast<int>(size) - 1);

    auto ai = static_cast<size_t>(activityIndex);
    do {
      auto job = route->tour.get(static_cast<std::size_t>(activityIndex))->job;

      if (job) return job;

      ai = (ai + 1) % size;

    } while (ai != activityIndex);

    return {};
  }
};
}