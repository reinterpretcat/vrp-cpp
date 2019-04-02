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

SCENARIO("single pickup and delivery can be solved", "[scenario][pickdev]") {
  GIVEN("problem with one shipment and one vehicle type") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}.addJob(build_test_shipment_job{}
                                         .id("job1")
                                         .demand(1)
                                         .pickup({
                                           {"location", json::array({1.0, 0.0})},
                                           {"duration", 10},
                                           {"times", LargeTimeWindows},
                                         })
                                         .delivery({{"location", json::array({2.0, 0.0})}, {"duration", 10}})
                                         .content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.id("vehicle").amount(1).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,1,1,0,2,1,2,0],
                        "durations": [0,1,1,1,0,2,1,2,0]
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
    "cost": 38.0,
    "distance": 4,
    "duration": 24,
    "times": {
      "driving": 4,
      "serving": 20,
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
          "load": [0],
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
            "departure": "1970-01-01T00:00:11Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "job1",
              "type": "pickup"
            }
          ]
        },
        {
          "location": [2.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:12Z",
            "departure": "1970-01-01T00:00:22Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:24Z",
            "departure": "1970-01-01T00:00:24Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "arrival",
              "type": "arrival"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 38.0,
        "distance": 4,
        "duration": 24,
        "times": {
          "driving": 4,
          "serving": 20,
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
