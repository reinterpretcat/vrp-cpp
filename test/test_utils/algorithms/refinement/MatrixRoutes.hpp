#pragma once

#include "models/costs/MatrixTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <memory>

namespace vrp::test {

/// Generates problem and solution which has routes distributed uniformly.
struct generate_matrix_routes final {
  using Result = std::pair<std::shared_ptr<vrp::models::Problem>, std::shared_ptr<vrp::models::Solution>>;

  Result operator()(int routeSize, int jobSize) const {
    using namespace vrp::models;
    using namespace vrp::models::costs;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace vrp::models::solution;

    auto fleet = std::make_shared<Fleet>();
    auto routes = std::vector<std::shared_ptr<Route>>{};
    auto jobs = std::set<Job, compare_jobs>{};

    ranges::for_each(ranges::view::ints(0, routeSize), [&](int i) {
      auto vehicle = test_build_vehicle{}.id(std::string("v" + std::to_string(i))).shared();
      auto actor = test_build_actor{}.vehicle(vehicle).shared();
      fleet->add(*vehicle);
      routes.push_back(test_build_route{}.actor(actor).shared());

      ranges::for_each(ranges::view::ints(0, jobSize), [&](int j) {
        int index = i * jobSize + j;
        auto job = as_job(test_build_service{}
                            .location(static_cast<Location>(index))
                            .id(std::string("c" + std::to_string(index)))
                            .shared());

        jobs.insert(job);
        routes[i]->tour.add(test_build_activity{}.job(job).shared());
      });
    });

    // TODO
    //    auto matrix = std::make_shared<MatrixTransportCosts>({}, {});
    //    auto problem = std::make_shared<Problem>({
    //        fleet,
    //        std::make_shared<Jobs>(*matrix, ranges::view::all(jobs), ranges::view::single("car")),
    //        {},
    //        matrix
    //    });
    return Result{};
  }
};
}