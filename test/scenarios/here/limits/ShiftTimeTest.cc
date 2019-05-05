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

SCENARIO("vehicle can be limited by shift time skipping job due duration", "[scenario][limits]") {
  GIVEN("problem with one job and one type") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}.addJob(build_test_delivery_job{}.content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.id("vehicle").amount(1).limits({}, {99}).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [1,1,1,1],
                        "durations": [1,100,100,1]
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
          "code": 102,
          "description": "cannot be assigned due to shift time constraint of vehicle"
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

SCENARIO("vehicle can be limited by shift time skipping job due to operation time and duration", "[scenario][limits]") {
  GIVEN("problem with one job and one type") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}
                            .addJob(build_test_delivery_job{}.id("job1").location(1, 0).duration(10).content())
                            .addJob(build_test_delivery_job{}.id("job2").location(2, 0).duration(10).content())
                            .addJob(build_test_delivery_job{}.id("job3").location(3, 0).duration(10).content())
                            .addJob(build_test_delivery_job{}.id("job4").location(4, 0).duration(10).content())
                            .addJob(build_test_delivery_job{}.id("job5").location(5, 0).duration(10).content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}  //
                                                           .id("vehicle")
                                                           .amount(1)
                                                           .capacity(5)
                                                           .limits({}, {40})
                                                           .content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,2,3,4,1,1,0,1,2,3,2,2,1,0,1,2,3,3,2,1,0,1,4,4,3,2,1,0,5,1,2,3,4,5,0],
                        "durations": [0,1,2,3,4,1,1,0,1,2,3,2,2,1,0,1,2,3,3,2,1,0,1,4,4,3,2,1,0,5,1,2,3,4,5,0]
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
    "cost": 52.0,
    "distance": 6,
    "duration": 36,
    "times": {
      "driving": 6,
      "serving": 30,
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
            3.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:03Z",
            "departure": "1970-01-01T00:00:13Z"
          },
          "load": [
            2
          ],
          "activities": [
            {
              "jobId": "job3",
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
            "arrival": "1970-01-01T00:00:14Z",
            "departure": "1970-01-01T00:00:24Z"
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
            1.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:25Z",
            "departure": "1970-01-01T00:00:35Z"
          },
          "load": [
            0
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
            0.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:36Z",
            "departure": "1970-01-01T00:00:36Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "arrival",
              "type": "arrival"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 52.0,
        "distance": 6,
        "duration": 36,
        "times": {
          "driving": 6,
          "serving": 30,
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
          "code": 102,
          "description": "cannot be assigned due to shift time constraint of vehicle"
        }
      ]
    },
    {
      "jobId": "job4",
      "reasons": [
        {
          "code": 102,
          "description": "cannot be assigned due to shift time constraint of vehicle"
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
