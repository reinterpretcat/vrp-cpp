#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"
#include "models/costs/ActivityCosts.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("empty route") {
    auto progress = test_build_insertion_progress{}.owned();
    auto constraint = std::make_shared<InsertionConstraint>();
    auto route = test_build_route{}.owned();
    auto evaluator = ServiceInsertionEvaluator(std::make_shared<TestTransportCosts>(),
        std::make_shared<ActivityCosts>(),
        constraint);

    bool failed = false;
    constraint->addHardRoute([&failed](const auto&, const auto&) {
      return failed ? InsertionConstraint::HardRouteResult{42} : InsertionConstraint::HardRouteResult{};
    });

    WHEN("service has failed constraint") {
      failed = true;
      auto result = evaluator.evaluate(ranges::get<0>(DefaultService),
                                       test_build_insertion_route_context{}.owned(), {});

      THEN("returns insertion failure") {
         REQUIRE (result.index() == 1);
      }

      THEN("returns failed constraint code") {
         REQUIRE (ranges::get<1>(result).constraint == 42);
      }
    }

    WHEN("service is ok") {
      auto result = evaluator.evaluate(ranges::get<0>(DefaultService),
                                       test_build_insertion_route_context{}.owned(),
                                       progress);

      THEN("returns insertion success") {
        REQUIRE (result.index() == 0);
      }

      THEN("returns correct index") {
        REQUIRE (ranges::get<0>(result).index == 0);
      }

      THEN("returns correct departure time") {
        REQUIRE (ranges::get<0>(result).departure == 0);
      }

      THEN("returns correct activity location") {
        REQUIRE (ranges::get<0>(result).activity->location == DefaultJobLocation);
      }
    }
  }
}

}