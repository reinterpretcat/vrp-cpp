#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/streams/HereModelBuilders.hpp"

#include <any>
#include <catch/catch.hpp>

using namespace nlohmann;
using namespace vrp;
using namespace vrp::models::problem;
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
                          .times(json::array({json::array({"1970-01-01T00:00:00Z", "1970-01-01T00:01:40Z"})}))
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
      THEN("has no unassigned") {
        // TODO
      }

      THEN("has two tours") {
        // TODO
      }

      THEN("has total statistic") {
        // TODO
      }

      THEN("has first tour statistic") {
        // TODO
      }

      THEN("has second tour statistic") {
        // TODO
      }
    }
  }
}
}
