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

SCENARIO("strict lock can be used with two locks for the same vehicle", "[scenario][relations]") {
  GIVEN("problem with two strict relations and one new job") {
    auto stream =
      build_test_problem{}
        .plan(
          build_test_plan{}
            .addJob(build_test_delivery_job{}.id("job1").location(1, 0).content())
            .addJob(build_test_delivery_job{}.id("job2").location(2, 0).content())
            .addJob(build_test_delivery_job{}.id("job3").location(3, 0).content())
            .addJob(build_test_delivery_job{}.id("job4").location(4, 0).content())
            .addJob(build_test_delivery_job{}.id("job5").location(5, 0).content())
            .addJob(build_test_delivery_job{}.id("job6").location(6, 0).content())
            .addJob(build_test_delivery_job{}.id("job7").location(7, 0).content())
            .addRelation(build_test_relation{}
                           .type("sequence")
                           .vehicle("vehicle_1")
                           .jobs({"departure", "job4", "job2", "job6"})
                           .content())
            .addRelation(build_test_relation{}.type("sequence").vehicle("vehicle_1").jobs({"job1", "job3"}).content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.id("vehicle").amount(1).capacity(7).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,2,3,4,5,6,1,1,0,1,2,3,4,5,2,2,1,0,1,2,3,4,3,3,2,1,0,1,2,3,4,4,3,2,1,0,1,2,5,5,4,3,2,1,0,1,6,6,5,4,3,2,1,0,7,1,2,3,4,5,6,7,0],
                        "durations": [0,1,2,3,4,5,6,1,1,0,1,2,3,4,5,2,2,1,0,1,2,3,4,3,3,2,1,0,1,2,3,4,4,3,2,1,0,1,2,5,5,4,3,2,1,0,1,6,6,5,4,3,2,1,0,7,1,2,3,4,5,6,7,0]
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
    "cost": 61.0,
    "distance": 22,
    "duration": 29,
    "times": {
      "driving": 22,
      "serving": 7,
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
          "load": [7],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [4.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:04Z",
            "departure": "1970-01-01T00:00:05Z"
          },
          "load": [6],
          "activities": [
            {
              "jobId": "job4",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [2.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:07Z",
            "departure": "1970-01-01T00:00:08Z"
          },
          "load": [5],
          "activities": [
            {
              "jobId": "job2",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [6.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:12Z",
            "departure": "1970-01-01T00:00:13Z"
          },
          "load": [4],
          "activities": [
            {
              "jobId": "job6",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [7.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:14Z",
            "departure": "1970-01-01T00:00:15Z"
          },
          "load": [3],
          "activities": [
            {
              "jobId": "job7",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [5.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:17Z",
            "departure": "1970-01-01T00:00:18Z"
          },
          "load": [2],
          "activities": [
            {
              "jobId": "job5",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [1.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:22Z",
            "departure": "1970-01-01T00:00:23Z"
          },
          "load": [1],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [3.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:25Z",
            "departure": "1970-01-01T00:00:26Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job3",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:29Z",
            "departure": "1970-01-01T00:00:29Z"
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
        "cost": 61.0,
        "distance": 22,
        "duration": 29,
        "times": {
          "driving": 22,
          "serving": 7,
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
