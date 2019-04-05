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

namespace {

auto defaultBreak = json({{"times", json::array({json::array({"1970-01-01T00:00:05Z", "1970-01-01T00:00:08Z"})})},
                          {"duration", 2},
                          {"location", json::array({6.0, 0.0})}});
}

namespace vrp::test::here {

SCENARIO("break can be skipped when vehicle is not used", "[scenarios][breaks]") {
  GIVEN("two jobs and two vehicles, one within break in between") {
    auto stream = build_test_problem{}
      .plan(build_test_plan{}
              .addJob(build_test_delivery_job{}.id("job1").location(5, 0).content())
              .addJob(build_test_delivery_job{}.id("job2").location(10, 0).duration(10).content()))
      .fleet(build_test_fleet{}
               .addVehicle(build_test_vehicle{}
                             .id("vehicle_with_break")
                             .locations({100, 0}, {100, 0})
                             .setBreak(defaultBreak)
                             .content())
               .addVehicle(build_test_vehicle{}.id("vehicle_without_break").amount(1).content()))
      .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,5,95,1,5,5,0,90,4,10,95,90,0,94,100,1,4,94,0,6,5,10,100,6,0],
                        "durations": [0,5,95,1,5,5,0,90,4,10,95,90,0,94,100,1,4,94,0,6,5,10,100,6,0]
                      })"_json}))
      .build();

    WHEN("solve problem") {
      auto estimatedSolution = SolverInstance(read_here_json_type{}(stream));

      THEN("vehicle without break is used and break is not considered as required job") {
        REQUIRE(estimatedSolution.first->routes.size() == 1);
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.second.actual == 61);
        REQUIRE(estimatedSolution.second.penalty == 0);

        REQUIRE(get_vehicle_id{}(*estimatedSolution.first->routes.front()->actor->vehicle) ==
          "vehicle_without_break_1");
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(*estimatedSolution.first),
                   Catch::Matchers::Equals(std::vector<std::string>{"job2", "job1"}));
      }
    }
  }
}

SCENARIO("break can be skipped when all jobs completed", "[scenarios][breaks]") {
  GIVEN("one jobs and vehicle with break") {
    auto stream = build_test_problem{}
      .plan(build_test_plan{}
              .addJob(build_test_delivery_job{}.id("job1").duration(10).content()))
      .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.setBreak(defaultBreak).amount(1).content()))
      .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,5,1,0,6,5,6,0],
                        "durations": [0,1,5,1,0,6,5,6,0]
                      })"_json}))
      .build();
    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);

      THEN("break is assigned") {
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
          "location": [1.0, 0.0],
          "time": {
            "arrival": "1970-01-01T00:00:01Z",
            "departure": "1970-01-01T00:00:11Z"
          },
          "load": [0],
          "activities": [
            {
              "jobId": "job1",
              "type": "delivery"
            }
          ]
        },
        {
          "location": [0.0, 0.0],
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
