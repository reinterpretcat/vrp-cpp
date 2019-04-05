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

SCENARIO("strict time windows should lead to two tours", "[scenario][times]") {
  GIVEN("problem with 5 jobs and two vehicles") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}
                          .id("job1")
                          .duration(10)
                          .location(10, 0)
                          .times(json::array({json::array({"1970-01-01T00:01:10Z", "1970-01-01T00:01:20Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job2")
                          .duration(10)
                          .location(20, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:50", "1970-01-01T00:01:00Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job3")
                          .duration(10)
                          .location(30, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:00:40Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job4")
                          .duration(10)
                          .location(40, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:00:40Z"})}))
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job5")
                          .duration(10)
                          .location(50, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:50Z", "1970-01-01T00:01:00Z"})}))
                          .content()))

        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.capacity(10).amount(2).content()))
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

      THEN("has two tours") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 390.0,
    "distance": 160,
    "duration": 210,
    "times": {
      "driving": 160,
      "serving": 50,
      "waiting": 0,
      "break": 0
    }
  },
  "tours": [
    {
      "vehicleId": "vehicle_2",
      "typeId": "vehicle",
      "stops": [
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:00Z",
            "departure": "1970-01-01T00:00:00Z"
          },
          "load": [3],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [30.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:30Z",
            "departure": "1970-01-01T00:00:40Z"
          },
          "load": [2],
          "activities": [
            {
              "jobId": "job3",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [20.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:50Z",
            "departure": "1970-01-01T00:01:00Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "job2",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [10.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:01:10Z",
            "departure": "1970-01-01T00:01:20Z"
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
            "arrival": "1970-01-01T00:01:30Z",
            "departure": "1970-01-01T00:01:30Z"
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
        "cost": 160.0,
        "distance": 60,
        "duration": 90,
        "times": {
          "driving": 60,
          "serving": 30,
          "waiting": 0,
          "break": 0
        }
      }
    },
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
          "load": [2],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [40.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:40Z",
            "departure": "1970-01-01T00:00:50Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "job4",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [50.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:01:00Z",
            "departure": "1970-01-01T00:01:10Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job5",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:02:00Z",
            "departure": "1970-01-01T00:02:00Z"
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
        "cost": 230.0,
        "distance": 100,
        "duration": 120,
        "times": {
          "driving": 100,
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
