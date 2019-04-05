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

SCENARIO("departure is flexible for late time windows", "[scenarios][breaks]") {
  GIVEN("one job with late time window") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}.addJob(
                      build_test_delivery_job{}
                        .times(json::array({json::array({"1970-01-01T00:00:10Z", "1970-01-01T00:00:20Z"})}))
                        .content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,1,0],
                        "durations": [0,1,1,0]
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
    "cost": 24.0,
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
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:09Z",
            "departure": "1970-01-01T00:00:09Z"
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
          "location": [1.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:10Z",
            "departure": "1970-01-01T00:00:20Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:21Z",
            "departure": "1970-01-01T00:00:21Z"
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
        "cost": 24.0,
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
  ]
}
)");
      }
    }
  }
}
}
