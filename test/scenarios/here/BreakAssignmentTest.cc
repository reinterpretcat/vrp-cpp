#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "streams/in/json/HereJson.hpp"
#include "test_utils/algorithms/construction/Results.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace vrp;
using namespace vrp::streams::in;
using namespace vrp::algorithms::refinement;

namespace {
auto solver = Solver<create_refinement_context<>,
                     select_best_solution,
                     ruin_and_recreate_solution<>,
                     GreedyAcceptance<>,
                     MaxIterationCriteria,
                     log_to_console>{};
}

namespace vrp::test {

// TODO add tests
// 1. break can be assigned between jobs
// 2. break should not be assigned at the beginning
// 3. break should not be assigned at the end
// 4. break should not be assigned or penalized if vehicle is not used

SCENARIO("break can be assigned between jobs", "[scenarios][break]") {
  GIVEN("two jobs and break in between") {
    std::stringstream ss;
    ss << R"(
{
  "id": "problem",
  "plan": {
    "jobs": [
      {
        "id": "job1",
        "places": {
          "delivery": {
            "location": [5.0, 0.0],
            "duration": 1,
            "times": [["1970-01-01T00:00:00Z", "1970-01-01T00:16:40Z"]]
          }
        },
        "demand": [1]
      },
      {
        "id": "job2",
        "places": {
          "delivery": {
            "location": [10.0, 0.0],
            "duration": 10,
            "times": [["1970-01-01T00:00:00Z", "1970-01-01T00:16:40Z"]]
          }
        },
        "demand": [1]
      }
    ]
  },
  "fleet": {
    "types": [
      {
        "id": "vehicle",
        "profile": "car",
        "costs": {
          "distance": 1.0,
          "time": 1.0,
          "fixed": 10.0
        },
        "places": {
          "start": {
            "time": "1970-01-01T00:00:00Z",
            "location": [0.0, 0.0]
          },
          "end": {
            "time": "1970-01-01T00:16:40Z",
            "location": [0.0, 0.0]
          }
        },
        "capacity": [2],
        "amount": 1,
        "break": {
          "times": [["1970-01-01T00:00:05Z", "1970-01-01T00:00:08Z"]],
          "duration": 2,
          "location": [6.0, 0.0]
        }
      }
    ]
  },
  "matrices": [
    {
      "profile": "car",
      "distances": [0, 5, 5, 1, 5, 0, 10, 4, 5, 10, 0, 6, 1, 4, 6, 0],
      "durations": [0, 5, 5, 1, 5, 0, 10, 4, 5, 10, 0, 6, 1, 4, 6, 0]
    }
  ]
}
    )";
    WHEN("solver problem") {
      auto estimatedSolution = solver(read_here_json_type{}(ss));

      THEN("break is assigned") {
        REQUIRE(estimatedSolution.first->routes.size() == 1);
        REQUIRE(estimatedSolution.first->unassigned.empty());
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(*estimatedSolution.first),
                   Catch::Matchers::Equals(std::vector<std::string>{"job1", "break", "job2"}));
      }
    }
  }
}
}
