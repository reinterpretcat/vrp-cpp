#pragma once

#include "models/Solution.hpp"
#include "models/extensions/solution/Helpers.hpp"
#include "utils/Random.hpp"

#include <optional>

namespace vrp::algorithms::refinement {

/// Selects random job within its route from list of routes.
struct select_random_job final {
  using ReturnType = std::optional<std::pair<std::shared_ptr<const models::solution::Route>, models::problem::Job>>;

  /// Returns no job if routes are empty or there is no single activity withing job.
  ReturnType operator()(const std::vector<std::shared_ptr<const models::solution::Route>>& routes,
                        utils::Random& random) const {
    if (routes.empty()) return ReturnType{};

    auto routeIndex = random.uniform<int>(0, static_cast<int>(routes.size()) - 1);

    auto ri = static_cast<size_t>(routeIndex);
    do {
      auto route = routes[routeIndex];

      if (route->tour.hasJobs()) {
        auto job = random_job(route, random);
        if (job) return ReturnType{std::make_pair(route, job.value())};
      }

      ri = (ri + 1) % routes.size();
    } while (ri != routeIndex);

    return ReturnType{};
  }

private:
  /// Selects random job from route.
  std::optional<models::problem::Job> random_job(const std::shared_ptr<const models::solution::Route>& route,
                                                 utils::Random& random) const {
    auto size = route->tour.count();
    if (size == 0) return {};

    auto activityIndex = random.uniform<int>(1, static_cast<int>(size));

    auto ai = static_cast<size_t>(activityIndex);
    do {
      auto job = models::solution::retrieve_job{}(*route->tour.get(static_cast<std::size_t>(activityIndex)));

      if (job) return job;

      ai = (ai + 1) % (size + 1);

    } while (ai != activityIndex);

    return {};
  }
};
}