#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/models/Factories.hpp"
#include "test_utils/fakes/TestActivityCosts.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][construction][insertion]") {
  GIVEN("insertable service with location") {
    auto route = test_build_route{}.owned();
    auto constraint = std::make_shared<InsertionConstraint>();
    auto evaluator = ServiceInsertionEvaluator(std::make_shared<TestTransportCosts>(),
        std::make_shared<TestActivityCosts>(),
        constraint);

    bool failed = false;
    constraint->addHardRoute([&failed](const auto&, const auto&) {
      return failed ? InsertionConstraint::HardRouteResult{42} : InsertionConstraint::HardRouteResult{};
    });

    WHEN("evaluate insertion context with empty tour and failed constraint") {
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
  }
}

}