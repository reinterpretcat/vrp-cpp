#include "algorithms/construction/InsertionConstraint.hpp"

#include "test_utils/algorithms/construction/constraints/Helpers.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;


namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][construction][insertion]") {
  GIVEN("insertion constraint and route constraints") {
    auto constraint = InsertionConstraint{};
    auto job = HardRouteConstraint::Job{};

    WHEN("all two hard route constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{}; }))
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{}; }))
                        .hard(InsertionRouteContext{}, job);

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one of all two hard route constraints is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{1}; }))
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{}; }))
                        .hard(InsertionRouteContext{}, job);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two hard route constraints are not fulfilled") {
      THEN("hard returns first code") {
        auto result = constraint
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{1}; }))
                        .addHardRoute(std::make_shared<HardRouteWrapper>(
                          [](const auto&, const auto&) { return HardRouteConstraint::Result{3}; }))
                        .hard(InsertionRouteContext{}, job);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two soft route constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result =
          constraint.addSoftRoute(std::make_shared<SoftRouteWrapper>([](const auto&, const auto&) { return 13.1; }))
            .addSoftRoute(std::make_shared<SoftRouteWrapper>([](const auto&, const auto&) { return 29.0; }))
            .soft(InsertionRouteContext{}, job);

        REQUIRE(result == 42.1);
      }
    }
  }

  GIVEN("insertion constraint with activity constraints") {
    auto constraint = InsertionConstraint{};

    WHEN("all two hard activity constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
                        .addHardActivity(std::make_shared<HardActivityWrapper>(
                          [](const auto&, const auto&) { return HardActivityConstraint::Result{}; }))
                        .addHardActivity(std::make_shared<HardActivityWrapper>(
                          [](const auto&, const auto&) { return HardActivityConstraint::Result{}; }))
                        .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(!result.has_value());
      }
    }

    WHEN("only one of two hard activity constraints is fulfilled") {
      THEN("hard returns value") {
        auto result = constraint
                        .addHardActivity(std::make_shared<HardActivityWrapper>([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 1}};
                        }))
                        .addHardActivity(std::make_shared<HardActivityWrapper>(
                          [](const auto&, const auto&) { return HardActivityConstraint::Result{}; }))
                        .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result.has_value());
        REQUIRE(result.value() == std::tuple<bool, int>{true, 1});
      }
    }

    WHEN("all two hard activity constraints are not fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
                        .addHardActivity(std::make_shared<HardActivityWrapper>([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 1}};
                        }))
                        .addHardActivity(std::make_shared<HardActivityWrapper>([](const auto&, const auto&) {
                          return HardActivityConstraint::Result{{true, 2}};
                        }))
                        .hard(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result.has_value());
        REQUIRE(result.value() == std::tuple<bool, int>{true, 1});
      }
    }

    WHEN("all two soft activity constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result =
          constraint  //
            .addSoftActivity(std::make_shared<SoftActivityWrapper>([](const auto&, const auto&) { return 13.1; }))
            .addSoftActivity(std::make_shared<SoftActivityWrapper>([](const auto&, const auto&) { return 29.0; }))
            .soft(InsertionRouteContext{}, InsertionActivityContext{});

        REQUIRE(result == 42.1);
      }
    }
  }
}
}
