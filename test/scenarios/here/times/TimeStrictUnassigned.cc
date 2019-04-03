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

SCENARIO("strict time windows should lead to unassigned job", "[scenario][times]") {
  GIVEN("problem with 5 jobs and one vehicle") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}
                          .id("job1")
                          .duration(0)
                          .location(10, 0)
                          .times(json::array({json::array({DefaultTimeStart, "1970-01-01T00:00:10Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job2")
                          .duration(0)
                          .location(20, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:10Z", "1970-01-01T00:00:20Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job3")
                          .duration(0)
                          .location(30, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:20Z", "1970-01-01T00:00:30Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job4")
                          .duration(0)
                          .location(40, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:30Z", "1970-01-01T00:00:40Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job5")
                          .duration(0)
                          .location(50, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:00:10Z"})}))
                          .content()))

        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.capacity(10).amount(1).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,10,20,30,40,10,10,0,10,20,30,20,20,10,0,10,20,30,30,20,10,0,10,40,40,30,20,10,0,50,10,20,30,40,50,0],
                        "durations": [0,10,20,30,40,10,10,0,10,20,30,20,20,10,0,10,20,30,30,20,10,0,10,40,40,30,20,10,0,50,10,20,30,40,50,0]
                      })"_json}))
        .build();

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("has empty solution with unassigned job within reason") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 170.0,
    "distance": 80,
    "duration": 80,
    "times": {
      "driving": 80,
      "serving": 0,
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
          "load": [4],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [10.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:10Z",
            "departure": "1970-01-01T00:00:10Z"
          },
          "load": [3],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [20.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:20Z",
            "departure": "1970-01-01T00:00:20Z"
          },
          "load": [2],
          "activities": [
            {
              "jobId": "job2",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [30.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:30Z",
            "departure": "1970-01-01T00:00:30Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "job3",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [40.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:40Z",
            "departure": "1970-01-01T00:00:40Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job4",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:01:20Z",
            "departure": "1970-01-01T00:01:20Z"
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
        "cost": 170.0,
        "distance": 80,
        "duration": 80,
        "times": {
          "driving": 80,
          "serving": 0,
          "waiting": 0,
          "break": 0
        }
      }
    }
  ],
  "unassigned": [
    {
      "jobId": "job5",
      "reasons": [
        {
          "code": 2,
          "description": "cannot be visited within time window"
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
