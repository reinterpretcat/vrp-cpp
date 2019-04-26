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

SCENARIO("strict lock can be used with tour relation and new job", "[scenario][relations]") {
  GIVEN("problem with tour relation and one new job") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}
                            .addJob(build_test_delivery_job{}.id("job1").location(1, 0).content())
                            .addJob(build_test_delivery_job{}.id("job2").location(2, 0).content())
                            .addJob(build_test_delivery_job{}.id("job3").location(3, 0).content())
                            .addRelation(build_test_relation{}  //
                                           .type("tour")
                                           .vehicle("vehicle_1")
                                           .jobs({"job1", "job3"})
                                           .content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}  //
                                                           .id("vehicle")
                                                           .amount(1)
                                                           .capacity(3)
                                                           .end(nlohmann::json{})
                                                           .content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,2,1,1,0,1,2,2,1,0,3,1,2,3,0],
                        "durations": [0,1,2,1,1,0,1,2,2,1,0,3,1,2,3,0]
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
    "cost": 19.0,
    "distance": 3,
    "duration": 6,
    "times": {
      "driving": 3,
      "serving": 3,
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
          "location": [
            0.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:00Z",
            "departure": "1970-01-01T00:00:00Z"
          },
          "load": [
            3
          ],
          "activities": [
            {
              "jobId": "departure",
              "type": "departure"
            }
          ]
        },
        {
          "location": [
            1.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:01Z",
            "departure": "1970-01-01T00:00:02Z"
          },
          "load": [
            2
          ],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [
            2.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:03Z",
            "departure": "1970-01-01T00:00:04Z"
          },
          "load": [
            1
          ],
          "activities": [
            {
              "jobId": "job2",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [
            3.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:05Z",
            "departure": "1970-01-01T00:00:06Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "job3",
              "type": "delivery"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 19.0,
        "distance": 3,
        "duration": 6,
        "times": {
          "driving": 3,
          "serving": 3,
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
