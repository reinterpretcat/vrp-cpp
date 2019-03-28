#include "Solver.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "streams/in/json/HereJson.hpp"
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

auto defaultBreak = json({{"times", json::array({json::array({"1970-01-01T00:00:05Z", "1970-01-01T00:00:08Z"})})},
                          {"duration", 2},
                          {"location", json::array({6.0, 0.0})}});
}

namespace vrp::test::here {

// TODO add tests
// 2. break should not be assigned at the beginning
// 3. break should not be assigned at the end

SCENARIO("break can be assigned between jobs", "[scenarios][break]") {
  GIVEN("two jobs and break in between") {
    auto stream = build_test_problem{}
                    .plan(build_test_plan{}
                            .addJob(build_test_delivery_job{}.id("job1").location(5, 0).content())
                            .addJob(build_test_delivery_job{}.id("job2").location(10, 0).content()))
                    .fleet(build_test_fleet{}.addVehicle(build_test_vehicle{}.setBreak(defaultBreak).content()))
                    .matrices(json::array({R"(
                      {
                        "profile": "car",
                        "distances": [0, 5, 5, 1, 5, 0, 10, 4, 5, 10, 0, 6, 1, 4, 6, 0],
                        "durations": [0, 5, 5, 1, 5, 0, 10, 4, 5, 10, 0, 6, 1, 4, 6, 0]
                      })"_json}))
                    .build();
    WHEN("solver problem") {
      auto estimatedSolution = solver(read_here_json_type{}(stream));

      THEN("break is assigned") {
        REQUIRE(estimatedSolution.first->routes.size() == 1);
        REQUIRE(estimatedSolution.first->unassigned.empty());
        CHECK_THAT(get_job_ids_from_all_routes{}.operator()(*estimatedSolution.first),
                   Catch::Matchers::Equals(std::vector<std::string>{"job1", "break", "job2"}));
      }
    }
  }
}

SCENARIO("break can be skipped when vehicle is not used", "[scenarios][break]") {
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

    WHEN("solver problem") {
      auto estimatedSolution = solver(read_here_json_type{}(stream));

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
}
