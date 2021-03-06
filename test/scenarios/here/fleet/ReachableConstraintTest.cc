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

SCENARIO("unreachable locations are handled as unreachable jobs", "[scenario][fleet]") {
  GIVEN("problem with one job and one type") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}.addJob(build_test_delivery_job{}.content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.id("vehicle").content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,2147483647,2147483647,0],
                        "durations": [0,1,1,0]
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
          "code": 100,
          "description": "location unreachable"
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
