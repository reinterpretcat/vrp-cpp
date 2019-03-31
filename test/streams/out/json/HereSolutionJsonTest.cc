#include "streams/out/json/HereSolutionJson.hpp"

#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "test_utils/streams/HereModelBuilders.hpp"

#include <catch/catch.hpp>
#include <sstream>

using namespace nlohmann;
using namespace vrp;
using namespace vrp::algorithms::refinement;
using namespace vrp::test::here;
using namespace vrp::models::problem;
using namespace vrp::streams::in;
using namespace vrp::streams::out;

namespace {

auto solver = Solver<create_refinement_context<>,
                     select_best_solution,
                     ruin_and_recreate_solution<>,
                     GreedyAcceptance<>,
                     MaxIterationCriteria,
                     log_to_console>{};
}

namespace vrp::test {

SCENARIO("can generate solution in here json format", "[streams][out][json][here]") {
  GIVEN("solution for problem with two jobs and one tour") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}
                            .addJob(build_test_delivery_job{}.id("job1").location(5, 0).content())
                            .addJob(build_test_delivery_job{}.id("job2").location(10, 0).content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.amount(1).content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,5,5,5,0,10,5,10,0],
                        "durations": [0,5,5,5,0,10,5,10,0]
                      })"_json}))
                    .build();
    auto problem = read_here_json_type{}(stream);
    auto estimatedSolution = solver(problem);

    WHEN("serialize solution as here json") {
      std::stringstream ss;
      dump_solution_as_here_json{problem}(ss, estimatedSolution);
      auto result = json::parse(ss.str());

      THEN("proper json is created") {
        REQUIRE(result == R"(
{
  "problemId": "problem",
  "statistic": {
    "cost": 52.0,
    "distance": 20,
    "duration": 22,
    "times": {
      "driving": 20,
      "serving": 2,
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
            10.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:10Z",
            "departure": "1970-01-01T00:00:11Z"
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
            5.0,
            0.0
          ],
          "time": {
            "arrival": "1970-01-01T00:00:16Z",
            "departure": "1970-01-01T00:00:17Z"
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
            "arrival": "1970-01-01T00:00:22Z",
            "departure": "1970-01-01T00:00:22Z"
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
        "distance": 20,
        "duration": 22,
        "times": {
          "driving": 20,
          "serving": 2,
          "waiting": 0,
          "break": 0
        }
      }
    }
  ]
}
          )"_json);
      }
    }
  }
}
}
