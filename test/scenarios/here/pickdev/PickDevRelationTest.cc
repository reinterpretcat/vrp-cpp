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

SCENARIO("pickup and delivery can be used in relation", "[scenario][pickdev]") {
  GIVEN("problem with two shipments in relation and one vehicle type") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}
                            .addJob(build_test_shipment_job{}
                                      .id("shipment1")
                                      .demand(1)
                                      .pickup({
                                        {"location", json::array({20.0, 0.0})},
                                        {"duration", 10},
                                        {"times", LargeTimeWindows},
                                      })
                                      .delivery({
                                        {"location", json::array({15.0, 0.0})},
                                        {"duration", 10},
                                        {"times", LargeTimeWindows},
                                      })
                                      .content())
                            .addJob(build_test_shipment_job{}
                                      .id("shipment2")
                                      .demand(1)
                                      .pickup({
                                                {"location", json::array({5.0, 0.0})},
                                                {"duration", 10},
                                                {"times", LargeTimeWindows},
                                              })
                                      .delivery({
                                                  {"location", json::array({20.0, 0.0})},
                                                  {"duration", 10},
                                                  {"times", LargeTimeWindows},
                                                })
                                      .content())
                            .addRelation(build_test_relation{}  //
                                           .type("sequence")
                                           .vehicle("vehicle_1")
                                           .jobs({"shipment1", "shipment2", "shipment1", "shipment2"})
                                           .content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}  //
                                                           .id("vehicle")
                                                           .amount(1)
                                                           .capacity(4)
                                                           .locations({10.0, 0.0}, {10.0, 0.0})
                                                           .content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,5,15,10,5,0,10,5,15,10,0,5,10,5,5,0],
                        "durations": [0,5,15,10,5,0,10,5,15,10,0,5,10,5,5,0]
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
    "cost": 150.0,
    "distance": 50,
    "duration": 90,
    "times": {
      "driving": 50,
      "serving": 40,
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
            10.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:00Z",
            "departure": "1970-01-01T00:00:00Z"
          },
          "load": [
            0
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
            20.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:10Z",
            "departure": "1970-01-01T00:00:20Z"
          },
          "load": [
            1
          ],
          "activities": [
            {
              "jobId": "shipment1",
              "type": "pickup"
            }
          ]
        },
        {
          "location": [
            5.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:35Z",
            "departure": "1970-01-01T00:00:45Z"
          },
          "load": [
            2
          ],
          "activities": [
            {
              "jobId": "shipment2",
              "type": "pickup"
            }
          ]
        },
        {
          "location": [
            15.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:55Z",
            "departure": "1970-01-01T00:01:05Z"
          },
          "load": [
            1
          ],
          "activities": [
            {
              "jobId": "shipment1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [
            20.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:01:10Z",
            "departure": "1970-01-01T00:01:20Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "shipment2",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [
            10.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:01:30Z",
            "departure": "1970-01-01T00:01:30Z"
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
        "cost": 150.0,
        "distance": 50,
        "duration": 90,
        "times": {
          "driving": 50,
          "serving": 40,
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
