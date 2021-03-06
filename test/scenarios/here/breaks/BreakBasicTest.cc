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
using namespace vrp::algorithms::refinement;

namespace {

auto defaultBreak = json({{"times", json::array({json::array({"1970-01-01T00:00:05Z", "1970-01-01T00:00:08Z"})})},
                          {"duration", 2},
                          {"location", json::array({6.0, 0.0})}});
}

namespace vrp::test::here {

// TODO add tests
// 2. break should not be assigned at the beginning
// 3. break should not be assigned at the end

SCENARIO("break can be assigned between jobs", "[scenarios][breaks]") {
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
    WHEN("solve problem") {
      auto estimatedSolution = SolverInstance(read_here_json_type{}(stream));

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
