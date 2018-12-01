#include "algorithms/construction/heuristics/CheapestInsertion.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;

namespace vrp::test {

SCENARIO("cheapest insertion inserts service", "[algorithms][construction][insertion]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used] = GENERATE(table<Location, EndLoc, EndLoc, std::string>({
    {3, {}, {}, "v1"},
    {21, {}, {}, "v2"},
  }));

  GIVEN("one service job and two vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());

    auto insertion = CheapestInsertion<InsertionEvaluator>{{std::make_shared<TestTransportCosts>(),
                                                            std::make_shared<ActivityCosts>(),
                                                            std::make_shared<InsertionConstraint>()}};

    WHEN("analyzes insertion context") {
      auto result = insertion.analyze(test_build_insertion_context{}
                                        .registry(std::make_shared<Registry>(fleet))
                                        .jobs({as_job(test_build_service{}.location(s1).shared())})
                                        .owned());

      THEN("returns new context with job inserted") {
        REQUIRE(result.unassigned.empty());
        REQUIRE(result.routes.size() == 1);
        REQUIRE(result.routes.begin()->first->actor->vehicle->id == used);
        REQUIRE(result.routes.begin()->first->tour.get(0)->detail.location == s1);
      }
    }
  }
}

SCENARIO("cheapest insertion handles c101_25 problem", "[algorithms][construction][insertion]") {
  // TODO
}

}
