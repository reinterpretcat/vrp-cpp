#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/scenarios/here/Assertions.hpp"
#include "test_utils/scenarios/here/Variables.hpp"
#include "test_utils/streams/HereModelBuilders.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace nlohmann;
using namespace vrp;
using namespace vrp::models::problem;
using namespace vrp::streams::in;

namespace vrp::test::here {

SCENARIO("single skills changes solution", "[scenario][skills]") {
  GIVEN("problem with one skilled job and two vehicles where only one has required skills") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}.addJob(build_test_delivery_job{}  //
                                         .skills({"unique_skill"})
                                         .content()))
        .fleet(build_test_fleet{}
                 .addVehicle(build_test_vehicle{}  //
                               .id("vehicle_without_skill")
                               .amount(1)
                               .content())
                 .addVehicle(build_test_vehicle{}
                               .id("vehicle_with_skill")
                               .amount(1)
                               .places({
                                 {"start", {{"time", "1970-01-01T00:00:00Z"}, {"location", json::array({10.0, 0.0})}}},
                                 {"end", {{"time", "1970-01-01T00:16:40Z"}, {"location", json::array({10.0, 0.0})}}},
                               })
                               .skills({"unique_skill"})
                               .content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,9,1,0,10,9,10,0],
                        "durations": [0,1,9,1,0,10,9,10,0]
                      })"_json}))
        .build();

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);
      auto jsonSolution = getSolutionAsJson(problem, estimatedSolution);

      THEN("uses more expensive vehicle but with skills") {
        REQUIRE(jsonSolution["tours"].size() == 1);
        REQUIRE(jsonSolution["tours"].front()["typeId"] == "vehicle_with_skill");
        REQUIRE(jsonSolution["statistic"] == R"(
{
    "cost": 47.0,
    "distance": 18,
    "duration": 19,
    "times": {
      "driving": 18,
      "serving": 1,
      "waiting": 0,
      "break": 0
    }
}
)"_json);
      }
    }
  }
}
}
