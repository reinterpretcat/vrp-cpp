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
using namespace vrp::algorithms::refinement;
using namespace vrp::test::here;

namespace {

auto defaultBreak = json({{"times", json::array({json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:01:40Z"})})},
                          {"duration", 10},
                          {"location", json::array({3.0, 0.0})}});

auto
getProblemStream(const std::string& relationType, std::initializer_list<std::string> jobs) {
  return build_test_problem{}
    .plan(build_test_plan{}
            .addJob(build_test_delivery_job{}.id("job1").location(1, 0).duration(10).content())
            .addJob(build_test_delivery_job{}.id("job2").location(2, 0).duration(10).content())
            .addRelation(build_test_relation{}  //
                           .type(relationType)
                           .vehicle("vehicle_1")
                           .jobs(jobs)
                           .content()))
    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}  //
                                           .setBreak(defaultBreak)
                                           .amount(1)
                                           .capacity(2)
                                           .end(nlohmann::json{})
                                           .content()))
    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,1,2,1,0,2,1,1,2,0,3,2,1,3,0],
                        "durations": [0,1,1,2,1,0,2,1,1,2,0,3,2,1,3,0]
                      })"_json}))
    .build();
}

}

namespace vrp::test::here {

SCENARIO("break with location can be used with relation (case 1)", "[scenarios][breaks]") {
  auto relationType = GENERATE(as<std::string>{}, "flexible", "sequence");

  GIVEN("two jobs and break in between") {
    auto stream = getProblemStream(relationType, {"departure", "job1", "break", "job2"});

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("has expected solution") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 48.0,
    "distance": 4,
    "duration": 34,
    "times": {
      "driving": 4,
      "serving": 20,
      "waiting": 0,
      "break": 10
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
            2
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
            "departure": "1970-01-01T00:00:11Z"
          },
          "load": [
            1
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
            3.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:13Z",
            "departure": "1970-01-01T00:00:23Z"
          },
          "load": [
            1
          ],
          "activities": [
            {
              "jobId": "break",
              "type": "break"
            }
          ]
        },
        {
          "location": [
            2.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:24Z",
            "departure": "1970-01-01T00:00:34Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "job2",
              "type": "delivery"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 48.0,
        "distance": 4,
        "duration": 34,
        "times": {
          "driving": 4,
          "serving": 20,
          "waiting": 0,
          "break": 10
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

SCENARIO("break with location can be used with relation (case 2)", "[scenarios][breaks]") {
  auto relationType = GENERATE(as<std::string>{}, "flexible", "sequence");

  GIVEN("two jobs and break in the end") {
    auto stream = getProblemStream(relationType, {"departure", "job1", "job2", "break"});

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("has expected solution") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 46.0,
    "distance": 3,
    "duration": 33,
    "times": {
      "driving": 3,
      "serving": 20,
      "waiting": 0,
      "break": 10
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
            2
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
            "departure": "1970-01-01T00:00:11Z"
          },
          "load": [
            1
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
            "arrival": "1970-01-01T00:00:12Z",
            "departure": "1970-01-01T00:00:22Z"
          },
          "load": [
            0
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
            "arrival": "1970-01-01T00:00:23Z",
            "departure": "1970-01-01T00:00:33Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "break",
              "type": "break"
            }
          ]
        }
      ],
      "statistic": {
        "cost": 46.0,
        "distance": 3,
        "duration": 33,
        "times": {
          "driving": 3,
          "serving": 20,
          "waiting": 0,
          "break": 10
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
