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

SCENARIO("vehicle with open end can be used", "[scenario][fleet]") {
  GIVEN("problem with one job and one type") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}.addJob(build_test_delivery_job{}.content()))
                    .fleet(build_test_fleet{}.addVehicle(
                      build_test_vehicle{}
                        .id("vehicle")
                        .amount(1)
                        .places({
                          {"start", {{"time", "1970-01-01T00:00:00Z"}, {"location", json::array({0.0, 0.0})}}},
                        })
                        .content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,1,0],
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
    "cost": 13.0,
    "distance": 1,
    "duration": 2,
    "times": {
      "driving": 1,
      "serving": 1,
      "waiting": 0,
      "break": 0
    }
  },
  "tours": [
    {
      "vehicleId": "vehicle_1",
      "typeId": "vehicle",
      "stops": [
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:00Z",
            "departure": "1970-01-01T00:00:00Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [1.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:01Z",
            "departure": "1970-01-01T00:00:02Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 13.0,
        "distance": 1,
        "duration": 2,
        "times": {
          "driving": 1,
          "serving": 1,
          "waiting": 0,
          "break": 0
        }
      }
    }
  ]
}

)");
      }
    }
  }
}
}
