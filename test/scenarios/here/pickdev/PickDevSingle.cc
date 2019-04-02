#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/scenarios/here/Variables.hpp"
#include "test_utils/streams/HereModelBuilders.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace nlohmann;
using namespace vrp;
using namespace vrp::models::problem;
using namespace vrp::streams::in;

namespace vrp::test::here {

SCENARIO("pickup and delivery can be solved", "[scenario][pickdev]") {
  GIVEN("problem with one shipment and one vehicle type") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}.addJob(build_test_shipment_job{}
                                                     .id("job1")
                                                     .demand(1)
                                                     .pickup({
                                                       {"location", json::array({1.0, 0.0})},
                                                       {"duration", 1},
                                                       {"times", LargeTimeWindows},
                                                     })
                                                     .delivery({{"location", json::array({2.0, 0.0})}, {"duration", 1}})
                                                     .content()))
                    .fleet(build_test_fleet{}.addVehicle(
                      build_test_vehicle{}
                        .id("vehicle")
                        .amount(1)
                        .places({
                          {"start", {{"time", DefaultTimeStart}, {"location", json::array({0.0, 0.0})}}},
                          {"end", {{"time", LargeTimeEnd}, {"location", json::array({0.0, 0.0})}}},
                        })
                        .capacity(10)
                        .costs(json({{"distance", 1.0}, {"time", 1.0}}))
                        .content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0,1,1,1,0,2,1,2,0],
                        "durations": [0,1,1,1,0,2,1,2,0]
                      })"_json}))
                    .build();

    WHEN("solve problem") {
      auto estimatedSolution = SolverInstance(read_here_json_type{}(stream));

      THEN("has expected tour") {
        // TODO
      }
    }
  }
}
}
