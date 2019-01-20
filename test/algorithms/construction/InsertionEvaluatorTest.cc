#include "algorithms/construction/InsertionEvaluator.hpp"

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::test;

namespace {

Tour::Activity
create(const std::string& id, int size) {
  return test_build_activity{}.job(as_job(test_build_service{}.id(id).dimens({{"size", size}}).shared())).shared();
}
}

namespace vrp::test {

SCENARIO("insertion evaluator can handle service insertion with time constraints",
         "[algorithms][construction][insertion]") {
  using EndLoc = std::optional<Location>;

  auto [s1, v1, v2, used, cost] = GENERATE(table<Location, EndLoc, EndLoc, std::string, Cost>({
    {3, {}, {}, "v1", (3 + 3) * 2},
    {27, {}, {}, "v2", (7 + 7) * 2},
    {11, {12}, {}, "v1", (12 + 12)},
  }));

  GIVEN("two different vehicles") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, v1, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{20, v2, {0, 100}}}).owned());
    auto constraint = std::make_shared<InsertionConstraint>();
    constraint->add<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(
      fleet, std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>()));

    auto evaluator = InsertionEvaluator{};

    WHEN("evaluates service insertion close to best vehicle") {
      auto service = test_build_service{}.location(s1).shared();

      auto result = evaluator.evaluate(
        as_job(service),
        test_build_insertion_context{}.constraint(constraint).registry(std::make_shared<Registry>(*fleet)).owned());

      THEN("returns correct insertion success") {
        REQUIRE(result.index() == 0);
        REQUIRE(ranges::get<0>(result).context.route->actor->vehicle->id == used);
        REQUIRE(ranges::get<0>(result).cost == cost);
      }
    }
  }
}
}
