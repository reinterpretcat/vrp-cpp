#include "algorithms/construction/insertion/InsertionConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][construction][insertion]") {
  GIVEN("insertion constraint and route constraints") {
    auto constraint = InsertionConstraint{};
    auto view = HardRouteConstraint::Activities{};

    WHEN("all two hard route constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint.addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{}; })
                        .addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{}; })
                        .hard(InsertionRouteContext{}, view);

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one of all two hard route constraints is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint.addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{1}; })
                        .addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{}; })
                        .hard(InsertionRouteContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two hard route constraints are not fulfilled") {
      THEN("hard returns first code") {
        auto result = constraint.addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{1}; })
                        .addHardRoute([](const auto&, const auto&) { return HardRouteConstraint::Result{3}; })
                        .hard(InsertionRouteContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two soft route constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result = constraint.addSoftRoute([](const auto&, const auto&) { return 13.1; })
                        .addSoftRoute([](const auto&, const auto&) { return 29.0; })
                        .soft(InsertionRouteContext{}, view);

        REQUIRE(result == 42.1);
      }
    }
  }

  GIVEN("insertion constraint with activity constraints") {
    auto constraint = InsertionConstraint{};

    WHEN("all two hard activity constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result =
          constraint.addHardActivity([](const auto&, const auto&) { return HardActivityConstraint::Result{}; })
            .addHardActivity([](const auto&, const auto&) { return HardActivityConstraint::Result{}; })
            .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(!result.has_value());
      }
    }

    WHEN("only one of two hard activity constraints is fulfilled") {
      THEN("hard returns value") {
        auto result = constraint  //
                        .addHardActivity([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 1}};
                        })
                        .addHardActivity([](const auto&, const auto&) { return HardActivityConstraint::Result{}; })
                        .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result.has_value());
        REQUIRE(result.value() == std::tuple<bool, int>{true, 1});
      }
    }

    WHEN("all two hard activity constraints are not fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint  //
                        .addHardActivity([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 1}};
                        })
                        .addHardActivity([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 2}};
                        })
                        .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result.has_value());
        REQUIRE(result.value() == std::tuple<bool, int>{true, 1});
      }
    }

    WHEN("all two soft activity constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result = constraint  //
                        .addSoftActivity([](const auto&, const auto&) { return 13.1; })
                        .addSoftActivity([](const auto&, const auto&) { return 29.0; })
                        .soft(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result == 42.1);
      }
    }
  }
}
}
