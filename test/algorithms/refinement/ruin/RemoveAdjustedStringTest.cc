#include "algorithms/refinement/ruin/RemoveAdjustedString.hpp"

#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/algorithms/refinement/MatrixRoutes.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models::problem;
using namespace vrp::utils;
using namespace Catch;

namespace {

/// Uses predefined values to control algorithm execution.
/// int distribution values:
/// 1. route index in solution
/// 2*. job index in selected route tour
/// 3*. selected algorithm: 1: sequential algorithm(**)
/// 4*. string removal index(-ies)
/// double distribution values:
/// 1. string count
/// 2*. string size(-s)
/// (*) - specific for each route.
/// (**) - calls more int and double distributions:
///     int 5. split start
///     dbl 3. alpha param
template<typename T>
struct FakeDistribution {
  std::vector<T> values = {};
  std::size_t index = 0;
  T operator()(T min, T max) {
    assert(index < values.size());
    auto value = values[index++];
    assert(value >= min && value <= max);
    return value;
  }
};
}

namespace vrp::test {

SCENARIO("adjusted string removal can ruin solution with single route", "[algorithms][refinement][ruin]") {
  auto [ints, doubles, ids] = GENERATE(table<std::vector<int>, std::vector<double>, std::vector<std::string>>({
    // sequential
    {{0, 3, 1, 1}, {1, 5}, {"c1", "c2", "c3", "c4", "c5"}},
    // preserved
    {{0, 2, 2, 0, 3}, {1, 5, 0.5, 0.005}, {"c0", "c1", "c2", "c5", "c6"}},
    {{0, 2, 2, 0, 3}, {1, 5, 0.5, 0.5, 0.005}, {"c0", "c1", "c2", "c6", "c7"}},
    {{0, 2, 2, 2, 3}, {1, 5, 0.5, 0.5, 0.005}, {"c2", "c6", "c7", "c8", "c9"}},
  }));

  GIVEN("solution with single route and 10 service jobs") {
    auto [problem, solution] = generate_matrix_routes{}(10, 1);

    WHEN("ruin without locked jobs") {
      auto context = RefinementContext{
        problem,
        std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
        std::make_shared<std::set<Job, compare_jobs>>()};

      RemoveAdjustedString{}.operator()(context, *solution);

      THEN("should ruin expected jobs") {
        CHECK_THAT(get_job_ids_from_set{}.operator()(solution->unassigned), Equals(ids));
      }
    }
  }
}

SCENARIO("adjusted string removal can ruin solution with multiple routes", "[algorithms][refinement][ruin]") {
  auto [ints, doubles, ids] = GENERATE(table<std::vector<int>, std::vector<double>, std::vector<std::string>>({
    // sequential
    {{1, 2, 1, 1}, {1, 3}, {"c6", "c7", "c8"}},
    {{0, 2, 1, 1, 1, 1, 2}, {2, 3, 2}, {"c1", "c2", "c3", "c7", "c8"}},
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {3, 3, 3, 3}, {"c1", "c11", "c12", "c13", "c2", "c3", "c6", "c7", "c8"}},
    // preserved
    {{1, 0, 2, 0, 2}, {1, 3, 0.5}, {"c5", "c6", "c9"}},
    {{1, 2, 2, 0, 2}, {1, 3, 0.5}, {"c5", "c6", "c7"}},
  }));

  GIVEN("solution with 3 routes within 5 service jobs in each") {
    auto [problem, solution] = generate_matrix_routes{}(5, 3);

    WHEN("ruin without locked jobs") {
      auto context = RefinementContext{
        problem,
        std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
        std::make_shared<std::set<Job, compare_jobs>>()};

      RemoveAdjustedString{}.operator()(context, *solution);

      THEN("should ruin expected jobs") {
        CHECK_THAT(get_job_ids_from_set{}.operator()(solution->unassigned), Equals(ids));
      }
    }
  }
}

SCENARIO("adjusted string removal can ruin solution using data generators", "[algorithms][refinement][ruin]") {
  auto jobs = GENERATE(range(10, 12));
  auto routes = GENERATE(range(1, 3));
  auto cardinality = GENERATE(range(5, 12));
  auto average = GENERATE(range(5, 12));
  auto alpha = GENERATE(values<double>({0.01, 0.1}));

  GIVEN("solution with service jobs") {
    auto [problem, solution] = generate_matrix_routes{}(jobs, routes);

    WHEN("ruin without locked jobs") {
      auto context =
        RefinementContext{problem, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>()};

      RemoveAdjustedString{cardinality, average, alpha}.operator()(context, *solution);

      THEN("should ruin some jobs") { REQUIRE(!solution->unassigned.empty()); }
    }
  }
}
}
