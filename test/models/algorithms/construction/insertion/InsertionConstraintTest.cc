#include "algorithms/construction/insertion/InsertionConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][construction][insertion]") {
  GIVEN("insertion constraint") {
    auto constraint = InsertionConstraint{};
    auto view = InsertionConstraint::Activities{};

    WHEN("all two hard route constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {}; })
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {}; })
            .hard(InsertionRouteContext{}, view);

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one of all two hard route constraints is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {1}; })
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {}; })
            .hard(InsertionRouteContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two hard route constraints are not fulfilled") {
      THEN("hard returns first code") {
        auto result = constraint
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {1}; })
            .addHardRoute([](const auto&, const auto&) { return InsertionConstraint::HardRouteResult {3}; })
            .hard(InsertionRouteContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two soft route constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result = constraint
            .addSoftRoute([](const auto&, const auto&) { return 13.1; })
            .addSoftRoute([](const auto&, const auto&) { return 29.0; })
            .soft(InsertionRouteContext{}, view);

        REQUIRE(result == 42.1);
      }
    }
  }
}

}
