#pragma once

#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "models/costs/ActivityCosts.hpp"
#include "models/costs/MatrixTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <cmath>
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
  static const inline std::string Profile = "car";

  using Result = std::pair<std::shared_ptr<vrp::models::Problem>, std::shared_ptr<vrp::models::Solution>>;

  Result operator()(int rows, int cols) const {
    using Matrix = vrp::models::costs::MatrixTransportCosts;
    using namespace ranges;
    using namespace vrp::models;
    using namespace vrp::models::common;
    using namespace vrp::models::problem;
    using namespace vrp::models::solution;

    auto fleet = std::make_shared<Fleet>();
    auto routes = std::vector<std::shared_ptr<const Route>>{};
    auto jobs = std::set<Job, compare_jobs>{};

    auto driver = test_build_driver{}.shared();
    fleet->add(*driver);

    ranges::for_each(ranges::view::ints(0, cols), [&](int i) {
      auto vehicle = test_build_vehicle{}.id(std::string("v" + std::to_string(i))).shared();
      auto actor = test_build_actor{}.driver(driver).vehicle(vehicle).shared();

      fleet->add(*actor->vehicle);
      routes.push_back(test_build_route{}.actor(actor).shared());

      ranges::for_each(ranges::view::ints(0, rows), [&](int j) {
        int index = i * rows + j;
        auto service = test_build_service{}
                         .location(static_cast<Location>(index))
                         .id(std::string("c" + std::to_string(index)))
                         .shared();

        jobs.insert(as_job(service));
        auto route = const_cast<Route*>(routes[i].get());
        route->tour.insert(test_build_activity{}.service(service).shared());
      });
    });

    auto registry = std::make_shared<Registry>(*fleet);
    ranges::for_each(routes, [&](const auto& route) { registry->use(route->actor); });

    auto matrix = std::make_shared<Matrix>(Matrix{values<Duration>(rows, cols), values<Distance>(rows, cols)});

    return Result{
      std::make_shared<Problem>(Problem{fleet,
                                        std::make_shared<Jobs>(*matrix, view::all(jobs), view::single(Profile)),
                                        std::make_shared<algorithms::construction::InsertionConstraint>(),
                                        std::make_shared<algorithms::objectives::penalize_unassigned_jobs<>>(),
                                        std::make_shared<costs::ActivityCosts>(),
                                        matrix}),
      std::make_shared<Solution>(Solution{std::move(registry), routes, std::map<Job, int, compare_jobs>{}})};
  }

private:
  template<typename T>
  std::map<std::string, std::vector<T>> values(int rows, int cols, T scale = 1000) const {
    auto size = cols * rows;
    auto sqr = [](T x) -> double { return static_cast<double>(x * x); };
    auto data = std::vector<T>(static_cast<size_t>(sqr(size)), 0);

    ranges::for_each(ranges::view::ints(0, size), [&](int i) {
      int left1 = i / rows, right1 = i % rows;
      ranges::for_each(ranges::view::ints(i + 1, size), [=, &data](int j) {
        auto [left2, right2] = std::make_pair(j / rows, j % rows);
        auto value = static_cast<T>(std::sqrt(sqr(left1 - left2) + sqr(right1 - right2)) * scale);

        data[i * size + j] = value;
        data[i * size + j + (j - i) * (size - 1)] = value;
      });
    });

    return {{Profile, data}};
  }
};
}