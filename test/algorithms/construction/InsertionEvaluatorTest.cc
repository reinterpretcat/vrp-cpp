#include "algorithms/construction/InsertionEvaluator.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;

using EndLoc = std::optional<Location>;

namespace vrp::test {

SCENARIO("insertion evaluator can handle service insertion", "[algorithms][construction][insertion]") {
  auto [s1, v1, v2, used, cost] = GENERATE(table<Location, EndLoc, EndLoc, std::string, Cost>({
    {3, {}, {}, "v1", (3 + 3) * 2},
    {27, {}, {}, "v2", (7 + 7) * 2},
    {11, {12}, {}, "v1", (12 + 12)},
  }));

  GIVEN("two different vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());

    auto evaluator = InsertionEvaluator{fleet,
                                        std::make_shared<TestTransportCosts>(),
                                        std::make_shared<ActivityCosts>(),
                                        std::make_shared<InsertionConstraint>()};

    WHEN("evaluates service insertion close to best vehicle") {
      auto service = test_build_service{}.location(s1).shared();

      auto result = evaluator.evaluate(as_job(service), test_build_insertion_context{}.owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).actor->vehicle->id == used);
        REQUIRE(ranges::get<0>(result).cost == cost);
      }
    }
  }
}
}