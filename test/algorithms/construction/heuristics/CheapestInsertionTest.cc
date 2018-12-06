#include "algorithms/construction/heuristics/CheapestInsertion.hpp"

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::streams::in;

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

    auto insertion = CheapestInsertion<InsertionEvaluator>{{std::make_shared<TestTransportCosts>(),
                                                            std::make_shared<ActivityCosts>(),
                                                            std::make_shared<InsertionConstraint>()}};

    WHEN("analyzes insertion context") {
      auto result = insertion.analyze(test_build_insertion_context{}
                                        .registry(std::make_shared<Registry>(fleet))
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

SCENARIO("cheapest insertion handles c101_25 problem", "[algorithms][construction][insertion]") {
  auto stream = create_c101_25_problem_stream{}();
  auto result = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);

  GIVEN("time and size constraints") {
    auto problem = std::get<0>(result);
    auto transportCosts = std::get<2>(result);
    auto activityCosts = std::get<1>(result);
    auto constraint = std::make_shared<InsertionConstraint>();
    (*constraint)
      .addHard<VehicleActivityTiming>(
        std::make_shared<VehicleActivityTiming>(problem.fleet, transportCosts, activityCosts))
      .addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());
    auto ctx = test_build_insertion_context{}
                 .jobs(std::move(problem.jobs))
                 .registry(std::make_shared<Registry>(problem.fleet))
                 .owned();

    THEN("calculates solution") {
      auto solution = CheapestInsertion<InsertionEvaluator>{{transportCosts, activityCosts, constraint}}.insert(ctx);

      REQUIRE(solution.jobs.empty());
      REQUIRE(solution.unassigned.empty());
      REQUIRE(!solution.routes.empty());
    }
  }
}
}
