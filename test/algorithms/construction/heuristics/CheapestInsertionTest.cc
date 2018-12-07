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
using namespace ranges;

namespace {

std::tuple<InsertionEvaluator, InsertionContext>
createInsertion(std::stringstream stream) {
  auto result = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);

  auto problem = std::get<0>(result);
  auto transport = std::get<2>(result);
  auto activity = std::get<1>(result);
  auto constraint = std::make_shared<InsertionConstraint>();
  (*constraint)
    .addHard<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(problem.fleet, transport, activity))
    .template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>())
    .addSoftRoute(std::make_shared<VehicleFixedCost>());
  auto ctx = vrp::test::test_build_insertion_context{}
               .jobs(std::move(problem.jobs))
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
  GIVEN("sequential coordinates problem with enough resources") {
    auto [vehicles, capacity, routes] = GENERATE(table<int, int, int>({
      {1, 10, 1},
      {2, 4, 2}
    }));

    auto [evaluator, ctx] = createInsertion<create_sequential_problem_stream>(vehicles, capacity);

    THEN("calculates solution with all jobs assigned") {
      auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.operator()(ctx);
      auto ids = get_job_ids_from_routes{}.operator()(solution);

      REQUIRE(solution.jobs.empty());
      REQUIRE(solution.unassigned.empty());
      REQUIRE(solution.routes.size() == routes);
    }
  }
}

SCENARIO("cheapest insertion handles solomon set problems", "[algorithms][construction][insertion]") {
  GIVEN("c101_25 problem") {
    auto [evaluator, ctx] = createInsertion<create_c101_25_problem_stream>();

    THEN("calculates solution") {
      auto solution = CheapestInsertion<InsertionEvaluator>{evaluator}.operator()(ctx);
      auto ids = get_job_ids_from_routes{}.operator()(solution);

      REQUIRE(solution.jobs.empty());
      REQUIRE(solution.unassigned.empty());
      REQUIRE(!solution.routes.empty());
      REQUIRE(solution.routes.size() == 4);
      REQUIRE(ranges::accumulate(ids, 0, [](const auto acc, const auto next) { return acc + 1; }) == 25);
    }
  }
}
}
