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

SCENARIO("waiting time is considered in solution", "[scenarios][breaks]") {
  GIVEN("two jobs, one of those has late time window") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}  //
                          .id("job1")
                          .duration(0)
                          .times(json::array({json::array({"1970-01-01T00:00:00", "1970-01-01T00:00:01"})}))
                          .location(1, 0)
                          .content())
                .addJob(build_test_delivery_job{}
                          .id("job2")
                          .duration(0)
                          .location(2, 0)
                          .times(json::array({json::array({"1970-01-01T00:00:10Z", "1970-01-01T00:00:20Z"})}))
                          .content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.amount(1).content()))
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

      THEN("has solution without waiting") {
        assertSolution(problem, estimatedSolution, R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 26.0,
    "distance": 4,
    "duration": 12,
    "times": {
      "driving": 4,
      "serving": 0,
      "waiting": 8,
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
          "load": [2],
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
            "departure": "1970-01-01T00:00:01Z"
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
          "location": [2.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:02Z",
            "departure": "1970-01-01T00:00:10Z"
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
            0.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:12Z",
            "departure": "1970-01-01T00:00:12Z"
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
        "cost": 26.0,
        "distance": 4,
        "duration": 12,
        "times": {
          "driving": 4,
          "serving": 0,
          "waiting": 8,
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
