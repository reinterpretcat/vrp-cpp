#include "algorithms/refinement/ruin/RemoveAdjustedString.hpp"

#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "streams/in/LiLim.hpp"
#include "test_utils/StreamSolver.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/algorithms/refinement/MatrixRoutes.hpp"
#include "test_utils/fakes/FakeDistribution.hpp"
#include "test_utils/streams/LiLimBuilder.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models::problem;
using namespace vrp::streams::in;
using namespace vrp::utils;
using namespace Catch;

namespace vrp::test {

/* Uses predefined values to control algorithm execution.
 int distribution values:
 1. route index in solution
 2*. job index in selected route tour
 3*. selected algorithm: 1: sequential algorithm(**)
 4*. string removal index(-ies)
 double distribution values:
 1. string count
 2*. string size(-s)
 (*) - specific for each route.
 (**) - calls more int and double distributions:
     int 5. split start
     dbl 3. alpha param
 */

// region Service

SCENARIO("adjusted string removal can ruin solution with single route", "[algorithms][refinement][ruin][service]") {
  auto [ints, doubles, ids] = GENERATE(table<std::vector<int>, std::vector<double>, std::vector<std::string>>({
    // sequential
    {{0, 3, 1, 2}, {1, 5}, {"c1", "c2", "c3", "c4", "c5"}},
    // preserved
    {{0, 2, 2, 1, 4}, {1, 5, 0.5, 0.005}, {"c0", "c1", "c2", "c5", "c6"}},
    {{0, 2, 2, 1, 4}, {1, 5, 0.5, 0.5, 0.005}, {"c0", "c1", "c2", "c6", "c7"}},
    {{0, 2, 2, 3, 4}, {1, 5, 0.5, 0.5, 0.005}, {"c2", "c6", "c7", "c8", "c9"}},
  }));

  GIVEN("solution with single route and 10 service jobs") {
    auto [problem, solution] = generate_matrix_routes{}(10, 1);

    WHEN("ruin without locked jobs") {
      auto context = RefinementContext{
        problem,
        std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
        std::make_shared<std::set<Job, compare_jobs>>(),
        {},
        0};

      auto iContext = restore_insertion_context{}(context, *solution);
      RemoveAdjustedString{}.operator()(context, *solution, iContext);

      THEN("should ruin expected jobs") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(iContext.solution->required), Equals(ids));
      }
    }
  }
}

SCENARIO("adjusted string removal can ruin solution with multiple routes", "[algorithms][refinement][ruin][service]") {
  auto [ints, doubles, ids] = GENERATE(table<std::vector<int>, std::vector<double>, std::vector<std::string>>({
    // sequential
    {{1, 2, 1, 2}, {1, 3}, {"c6", "c7", "c8"}},
    {{0, 2, 1, 2, 1, 3, 2}, {2, 3, 2}, {"c1", "c2", "c3", "c7", "c8"}},
    {{1, 1, 1, 2, 1, 2, 1, 2, 1, 2}, {3, 3, 3, 3}, {"c1", "c11", "c12", "c13", "c2", "c3", "c6", "c7", "c8"}},
    // preserved
    {{1, 1, 2, 1, 3}, {1, 3, 0.5}, {"c5", "c6", "c9"}},
    {{1, 3, 2, 1, 3}, {1, 3, 0.5}, {"c5", "c6", "c7"}},
  }));

  GIVEN("solution with 3 routes within 5 service jobs in each") {
    auto [problem, solution] = generate_matrix_routes{}(5, 3);

    WHEN("ruin without locked jobs") {
      auto context = RefinementContext{
        problem,
        std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
        std::make_shared<std::set<Job, compare_jobs>>(),
        {},
        0};
      auto iContext = restore_insertion_context{}(context, *solution);

      RemoveAdjustedString{}.operator()(context, *solution, iContext);

      THEN("should ruin expected jobs") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(iContext.solution->required), Equals(ids));
      }
    }
  }
}

SCENARIO("adjusted string removal can ruin solution using data generators", "[algorithms][refinement][ruin][service]") {
  auto jobs = GENERATE(range(10, 12));
  auto routes = GENERATE(range(1, 3));
  auto cardinality = GENERATE(range(5, 12));
  auto average = GENERATE(range(5, 12));
  auto alpha = GENERATE(values<double>({0.01, 0.1}));

  GIVEN("solution with service jobs") {
    auto [problem, solution] = generate_matrix_routes{}(jobs, routes);

    WHEN("ruin without locked jobs") {
      auto context =
        RefinementContext{problem, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>(), {}, 0};

      auto iCtx = restore_insertion_context{}(context, *solution);

      RemoveAdjustedString{cardinality, average, alpha}.operator()(context, *solution, iCtx);

      THEN("should ruin some jobs and remove empty tours") {
        REQUIRE(!iCtx.solution->required.empty());
        REQUIRE(ranges::accumulate(
          iCtx.solution->routes, true, [](bool acc, const auto& pair) { return acc && pair.route->tour.hasJobs(); }));
      }
    }
  }
}

// endregion

SCENARIO("adjusted string removal can ruin solution with sequence", "[algorithms][refinement][ruin][sequence]") {
  auto [ints, doubles, removed, kept] =
    GENERATE(table<std::vector<int>, std::vector<double>, std::vector<std::string>, std::vector<std::string>>({
      // sequential
      {{0, 3, 1, 2}, {1, 3}, {"seq2", "seq3"}, {"seq1", "seq1", "seq0", "seq0"}},
      // preserved
      {{0, 2, 2, 3, 4}, {1, 2, 1, 0.005}, {"seq1", "seq2"}, {"seq3", "seq3", "seq0", "seq0"}},
      {{0, 2, 2, 1, 2}, {1, 2, 0.001, 0.001}, {"seq2", "seq3"}, {"seq1", "seq1", "seq0", "seq0"}},
    }));

  GIVEN("reference problem with four sequences") {
    struct create_reference_problem_stream {
      std::stringstream operator()(int vehicles = 1, int capacity = 200) {
        return LiLimBuilder()
          .setVehicle(vehicles, capacity)
          .addCustomer({0, 0, 0, 0, 0, 1000, 0, 0, 0})
          .addCustomer({1, 1, 0, -1, 0, 1000, 0, 2, 0})
          .addCustomer({2, 2, 0, 1, 0, 1000, 0, 0, 1})
          .addCustomer({3, 3, 0, -1, 0, 1000, 0, 4, 0})
          .addCustomer({4, 4, 0, 1, 0, 1000, 0, 0, 3})
          .addCustomer({5, 5, 0, -1, 0, 1000, 0, 6, 0})
          .addCustomer({6, 6, 0, 1, 0, 1000, 0, 0, 5})
          .addCustomer({7, 7, 0, -1, 0, 1000, 0, 8, 0})
          .addCustomer({8, 8, 0, 1, 0, 1000, 0, 0, 7})
          .build();
      }
    };

    auto [problem, solution] = solve_stream<create_reference_problem_stream, read_li_lim_type<cartesian_distance>>{}();
    auto ctx =
      RefinementContext{problem,
                        std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
                        std::make_shared<std::set<Job, compare_jobs>>(),
                        {},
                        0};

    WHEN("ruin without locked jobs") {
      auto iCtx = restore_insertion_context{}(ctx, *solution);

      RemoveAdjustedString{}.operator()(
        RefinementContext{
          problem,
          std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{doubles, 0}),
          std::make_shared<std::set<Job, compare_jobs>>(),
          {},
          0},
        *solution,
        iCtx);

      THEN("should ruin expected jobs") {
        CHECK_THAT(get_job_ids_from_jobs{}.operator()(iCtx.solution->required), Equals(removed));
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(iCtx), Equals(kept));
      }
    }
  }
}
}
