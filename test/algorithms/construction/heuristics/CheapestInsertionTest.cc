#include "algorithms/construction/heuristics/CheapestInsertion.hpp"

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "algorithms/construction/constraints/VehicleFixedCost.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Extensions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;
using namespace Catch::Matchers;
using namespace ranges;

namespace {

std::tuple<InsertionEvaluator, InsertionContext>
createInsertion(std::stringstream stream) {
  auto problem = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);

  auto transport = problem.transport;
  auto activity = problem.activity;
  auto constraint = std::make_shared<InsertionConstraint>();
  (*constraint)
    .addHard<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(problem.fleet, transport, activity))
    .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>())
    .addSoftRoute(std::make_shared<VehicleFixedCost>());
  auto ctx = vrp::test::test_build_insertion_context{}
               .jobs(problem.jobs.all())
               .registry(std::make_shared<Registry>(problem.fleet))
               .constraint(constraint)
               .owned();

  return {{transport, activity}, ctx};
}

template<typename ProblemStream>
std::tuple<InsertionEvaluator, InsertionContext>
createInsertion(int vehicles, int capacities) {
  return createInsertion(ProblemStream{}(vehicles, capacities));
}

template<typename ProblemStream>
std::tuple<InsertionEvaluator, InsertionContext>
createInsertion() {
  return createInsertion(ProblemStream{}());
}
}

namespace vrp::test {

SCENARIO("cheapest insertion inserts service", "[algorithms][construction][insertion]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used] = GENERATE(table<Location, EndLoc, EndLoc, std::string>({
    {3, {}, {}, "v1"},
    {21, {}, {}, "v2"},
  }));

  GIVEN("one service job and two vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());

    auto insertion = CheapestInsertion<InsertionEvaluator>{
      {std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>()}};

    WHEN("analyzes insertion context") {
      auto result = insertion(test_build_insertion_context{}
                                .registry(std::make_shared<Registry>(fleet))
                                .constraint(std::make_shared<InsertionConstraint>())
                                .jobs({as_job(test_build_service{}.location(s1).shared())})
                                .owned());

      THEN("returns new context with job inserted") {
        REQUIRE(result.unassigned.empty());
        REQUIRE(result.routes.size() == 1);
        REQUIRE(result.routes.begin()->first->actor->vehicle->id == used);
        REQUIRE(result.routes.begin()->first->tour.get(0)->detail.location == s1);
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with demand", "[algorithms][construction][insertion]") {
  //  auto [vehicles, capacity, unassigned, routes] =
  //    GENERATE(table<int, int, int, int>({{1, 10, 0, 1}, {2, 4, 0, 2}, {1, 4, 1, 1}, {1, 3, 2, 1}}));
  // TODO what is wrong with generator here?
  for (auto [vehicles, capacity, unassigned, routes] :
       std::vector<std::tuple<int, int, int, int>>{{1, 10, 0, 1}, {2, 4, 0, 2}, {1, 4, 1, 1}, {1, 3, 2, 1}}) {
    GIVEN("sequential coordinates problem") {
      auto [evaluator, ctx] = createInsertion<create_sequential_problem_stream>(vehicles, capacity);

      WHEN("calculates solution") {
        auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.operator()(ctx);
        THEN("all jobs processed") {
          REQUIRE(solution.jobs.empty());
          REQUIRE(solution.unassigned.size() == unassigned);
          REQUIRE(solution.routes.size() == routes);
        }
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with times", "[algorithms][construction][insertion]") {
  GIVEN("time problem") {
    auto [evaluator, ctx] = createInsertion<create_time_problem_stream>(1, 10);

    WHEN("calculates solution") {
      auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.insert(ctx);

      THEN("all jobs processed") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        REQUIRE(get_job_ids_from_all_routes{}.operator()(solution).front() == "c5");
      }
    }
  }
}

SCENARIO("cheapest insertion handles artificial problems with waiting", "[algorithms][construction][insertion]") {
  GIVEN("time problem") {
    struct create_waiting_problem_stream {
      std::stringstream operator()() {
        return SolomonBuilder()
          .setVehicle(1, 10)
          .addCustomer({0, 0, 0, 0, 0, 1000, 0})
          .addCustomer({1, 1, 0, 1, 20, 40, 10})
          .addCustomer({2, 2, 0, 1, 50, 100, 10})
          .build();
      }
    };
    auto [evaluator, ctx] = createInsertion<create_waiting_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.insert(ctx);

      THEN("all jobs processed") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(solution.routes.size() == 1);
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(solution), Equals(std::vector<std::string>{"c1", "c2"}));
      }
    }
  }
}

SCENARIO("cheapest insertion handles solomon set problems", "[algorithms][construction][insertion]") {
  GIVEN("c101_25 problem") {
    auto [evaluator, ctx] = createInsertion<create_c101_25_problem_stream>();

    WHEN("calculates solution") {
      auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.operator()(ctx);
      auto ids = get_job_ids_from_all_routes{}.operator()(solution);

      THEN("has expected solution") {
        REQUIRE(solution.jobs.empty());
        REQUIRE(solution.unassigned.empty());
        REQUIRE(!solution.routes.empty());
        REQUIRE(solution.routes.size() == 6);
        REQUIRE(ranges::accumulate(ids, 0, [](const auto acc, const auto next) { return acc + 1; }) == 25);
      }
    }
  }
}
}
