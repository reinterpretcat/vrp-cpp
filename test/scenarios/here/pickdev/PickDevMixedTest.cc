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
using namespace vrp::test::here;

namespace {
const auto TestShipment = build_test_shipment_job{}
                            .id("job2")
                            .pickup({
                              {"location", json::array({2.0, 0.0})},
                              {"duration", 1},
                              {"times", LargeTimeWindows},
                            })
                            .delivery({{"location", json::array({3.0, 0.0})}, {"duration", 1}})
                            .content();
}

namespace vrp::test::here {

SCENARIO("pickup and delivery can be mixed with delivery jobs", "[scenario][pickdev]") {
  GIVEN("problem with 2 deliveries, one shipment and one vehicle type") {
    auto stream =
      build_test_problem{}
        .plan(build_test_plan{}
                .addJob(build_test_delivery_job{}.id("job1").location(1, 0).times(LargeTimeWindows).content())
                .addJob(TestShipment)
                .addJob(build_test_delivery_job{}.id("job3").location(4, 0).content()))
        .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}
                                               .id("vehicle")
                                               .amount(1)
                                               .capacity(10)
                                               .costs(json({{"distance", 1.0}, {"time", 1.0}}))
                                               .content()))
        .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,2,3,1,1,0,1,2,2,2,1,0,1,3,3,2,1,0,4,1,2,3,4,0],
                        "durations": [0,1,2,3,1,1,0,1,2,2,2,1,0,1,3,3,2,1,0,4,1,2,3,4,0]
                      })"_json}))
        .build();

    WHEN("solve problem") {
      auto problem = read_here_json_type{}(stream);
      auto estimatedSolution = SolverInstance(problem);
      auto jsonSolution = getSolutionAsJson(problem, estimatedSolution);

      THEN("has expected solution parameters") {
        assertShipment(TestShipment, jsonSolution["tours"].at(0));
        REQUIRE(jsonSolution["tours"].size() == 1);
        REQUIRE(jsonSolution["statistic"] == R"(
{
    "cost": 20.0,
    "distance": 8,
    "duration": 12,
    "times": {
      "driving": 8,
      "serving": 4,
      "waiting": 0,
      "break": 0
    }
}
)"_json);
      }
    }
  }
}
}
