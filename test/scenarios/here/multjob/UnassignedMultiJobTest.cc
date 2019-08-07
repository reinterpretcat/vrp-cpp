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

namespace vrp::test::here {

SCENARIO("Can unassign properly multi job", "[scenarios][multijob]") {
  GIVEN("having multi job and simple delivery") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}.id("simple").duration(10).location(1, 0).demand(1).content())
                .addJob(build_test_multi_job{}
                          .id("multi")
                          .addPickup(build_test_pickup_job{}.duration(10).location(2, 0).demand(2).tag("1"))
                          .addPickup(build_test_pickup_job{}.duration(10).location(8, 0).demand(1).tag("2"))
                          .addDelivery(build_test_delivery_job{}.duration(10).location(6, 0).demand(3).tag("3"))
                          .content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.amount(1).capacity(2).content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,7,5,1,1,0,6,4,2,7,6,0,2,8,5,4,2,0,6,1,2,8,6,0],
                        "durations": [0,1,7,5,1,1,0,6,4,2,7,6,0,2,8,5,4,2,0,6,1,2,8,6,0]
                      })"_json}))
        .build();
    WHEN("should have multi job unassigned") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("has solution without waiting") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 24,
    "distance": 2,
    "duration": 12,
    "times": {
      "driving": 2,
      "serving": 10,
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
            0,
            0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:00Z",
            "departure": "1970-01-01T00:00:00Z"
          },
          "load": [
            1
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
            1,
            0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:01Z",
            "departure": "1970-01-01T00:00:11Z"
          },
          "load": [
            0
          ],
          "activities": [
            {
              "jobId": "simple",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [
            0,
            0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:12Z",
            "departure": "1970-01-01T00:00:12Z"
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
        "cost": 24,
        "distance": 2,
        "duration": 12,
        "times": {
          "driving": 2,
          "serving": 10,
          "waiting": 0,
          "break": 0
        }
      }
    }
  ],
  "unassigned": [
    {
      "jobId": "multi",
      "reasons": [
        {
          "code": 3,
          "description": "does not fit into any vehicle due to capacity"
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
