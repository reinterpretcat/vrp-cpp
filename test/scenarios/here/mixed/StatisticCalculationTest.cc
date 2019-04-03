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

SCENARIO("statistic can be calculated for two simple tours", "[scenario][statistic]") {
  GIVEN("problem with 3 jobs and two vehicle types") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}
                          .id("job1")
                          .demand(1)
                          .location(0, 0)
                          .duration(10)
                          .times(SmallTimeWindows)
                          .content())
                .addJob(build_test_shipment_job{}
                          .id("job2")
                          .demand(1)
                          .pickup({
                            {"location", json::array({1.0, 0.0})},
                            {"duration", 10},
                            {"times", json::array({json::array({"1970-01-01T00:03:20Z", "1970-01-01T00:16:40Z"})})},
                          })
                          .delivery({{"location", json::array({2.0, 0.0})}, {"duration", 20}})
                          .content())
                .addJob(build_test_delivery_job{}  //
                          .id("job3")
                          .demand(1)
                          .location(3, 1)
                          .duration(5)
                          .skills({"my_skill"})
                          .content()))
        .fleet(build_test_fleet{}
                 .addVehicle(
                   build_test_vehicle{}
                     .id("vehicle1")
                     .amount(1)
                     .places({
                       {"start", {{"time", "1970-01-01T00:00:00Z"}, {"location", json::array({1.0, 0.0})}}},
                     })
                     .capacity(2)
                     .costs(json({{"distance", 1.0}, {"time", 2.0}}))
                     .setBreak({{"times", json::array({json::array({"1970-01-01T00:01:40Z", "1970-01-01T00:02:30Z"})})},
                                {"duration", 50}})
                     .content())
                 .addVehicle(build_test_vehicle{}
                               .id("vehicle2")
                               .amount(1)
                               .places({
                                 {"start", {{"time", "1970-01-01T00:00:00Z"}, {"location", json::array({0.0, 1.0})}}},
                                 {"end", {{"time", "1970-01-01T00:16:40Z"}, {"location", json::array({1.0, 1.0})}}},
                               })
                               .capacity(10)
                               .skills({"my_skill"})
                               .costs({{"distance", 2.0}, {"time", 4.0}, {"fixed", 100.0}})
                               .content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        "durations": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
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
              "cost": 601.0,
                "distance": 7,
                "duration": 239,
                "times": {
                "driving": 7,
                  "serving": 45,
                  "waiting": 137,
                  "break": 50
              }
            },
            "tours": [
            {
              "vehicleId": "vehicle_1",
                "typeId": "vehicle",
                "stops": [
              {
                "location": [1.0, 0.0],
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
                "location": [0.0, 0.0],
                "time": {
                  "arrival": "1970-01-01T00:00:01Z",
                    "departure": "1970-01-01T00:02:30Z"
                },
                "load": [0],
                "activities": [
                {
                  "jobId": "job1",
                    "type": "delivery",
                    "location": [0.0, 0.0],
                  "time": {
                    "start": "1970-01-01T00:00:01Z",
                      "end": "1970-01-01T00:00:11Z"
                  }
                },
                {
                  "jobId": "break",
                    "type": "break",
                    "location": [0.0, 0.0],
                  "time": {
                    "start": "1970-01-01T00:00:12Z",
                      "end": "1970-01-01T00:02:30Z"
                  }
                }
                ]
              },
              {
                "location": [1.0, 0.0],
                "time": {
                  "arrival": "1970-01-01T00:02:31Z",
                    "departure": "1970-01-01T00:03:30Z"
                },
                "load": [1],
                "activities": [
                {
                  "jobId": "job2",
                    "type": "pickup"
                }
                ]
              },
              {
                "location": [2.0, 0.0],
                "time": {
                  "arrival": "1970-01-01T00:03:31Z",
                    "departure": "1970-01-01T00:03:51Z"
                },
                "load": [0],
                "activities": [
                {
                  "jobId": "job2",
                    "type": "delivery"
                }
                ]
              }
              ],
              "statistic": {
                "cost": 469.0,
                  "distance": 5,
                  "duration": 232,
                  "times": {
                  "driving": 5,
                    "serving": 40,
                    "waiting": 137,
                    "break": 50
                }
              }
            },
            {
              "vehicleId": "myVehicle2_1",
                "typeId": "myVehicle2",
                "stops": [
              {
                "location": [0.0, 1.0],
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
                "location": [3.0, 1.0],
                "time": {
                  "arrival": "1970-01-01T00:00:01Z",
                    "departure": "1970-01-01T00:00:06Z"
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
                "location": [1.0, 1.0],
                "time": {
                  "arrival": "1970-01-01T00:00:07Z",
                    "departure": "1970-01-01T00:00:07Z"
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
                "cost": 132.0,
                  "distance": 2,
                  "duration": 7,
                  "times": {
                  "driving": 2,
                    "serving": 5,
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
