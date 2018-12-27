#pragma once

#include "models/costs/ActivityCosts.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <memory>

namespace vrp::test {

/// Generates problem and solution which has routes distributed uniformly, e.g.:
/// r0 r1 r2 r3
/// -----------
/// 0  4   8 12
/// 1  5   9 13
/// 2  6  10 14
/// 3  7  11 15
struct generate_matrix_routes final {
  using Result = std::pair<std::shared_ptr<vrp::models::Problem>, std::shared_ptr<vrp::models::Solution>>;

  Result operator()(int rows, int cols) const {
    using Matrix = vrp::models::costs::MatrixTransportCosts;
    using namespace ranges;
    using namespace vrp::models;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace vrp::models::solution;

    auto fleet = std::make_shared<Fleet>();
    auto routes = std::vector<std::shared_ptr<Route>>{};
    auto jobs = std::set<Job, compare_jobs>{};

    ranges::for_each(ranges::view::ints(0, cols), [&](int i) {
      auto vehicle = test_build_vehicle{}.id(std::string("v" + std::to_string(i))).shared();
      auto actor = test_build_actor{}.vehicle(vehicle).shared();
      fleet->add(*vehicle);
      routes.push_back(test_build_route{}.actor(actor).shared());

      ranges::for_each(ranges::view::ints(0, rows), [&](int j) {
        int index = i * rows + j;
        auto job = as_job(test_build_service{}
                            .location(static_cast<Location>(index))
                            .id(std::string("c" + std::to_string(index)))
                            .shared());

        jobs.insert(job);
        routes[i]->tour.add(test_build_activity{}.job(job).shared());
      });
    });

    auto matrix = std::make_shared<Matrix>(Matrix{values<Duration>(rows, cols), values<Distance>(rows, cols)});

    return Result{
      std::make_shared<Problem>(Problem{fleet,
                                        std::make_shared<Jobs>(*matrix, view::all(jobs), view::single("car")),
                                        std::make_shared<algorithms::construction::InsertionConstraint>(),
                                        std::make_shared<costs::ActivityCosts>(),
                                        matrix}),
      std::make_shared<Solution>(Solution{0, routes, std::set<Job, compare_jobs>{}})};
  }

private:
  template<typename T>
  std::map<std::string, std::vector<T>>&& values(int columns, int rows) const {
    // TODO
  }
};
}