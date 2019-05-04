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

SCENARIO("vehicle can be limited by max distance", "[scenario][limits]") {
  GIVEN("problem with one job and one type") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}.addJob(build_test_delivery_job{}.content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.id("vehicle").amount(1).limits({99}, {}).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [1,100,100,1],
                        "durations": [1,1,1,1]
                      })"_json}))
        .build();

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("has expected solution") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 0.0,
    "distance": 0,
    "duration": 0,
    "times": {
      "driving": 0,
      "serving": 0,
      "waiting": 0,
      "break": 0
    }
  },
  "tours": [],
  "unassigned": [
    {
      "jobId": "job1",
      "reasons": [
        {
          "code": 101,
          "description": "cannot be assigned due to max distance constraint of vehicle"
        }
      ]
    }
  ]
}
)");
      }
    }
  }
}
}
