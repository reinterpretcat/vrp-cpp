#include "algorithms/construction/insertion/InsertionConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][constraints]") {
  GIVEN("insertion constraint") {
    auto constraint = InsertionConstraint{};
    auto view = ranges::view::empty<const models::solution::Activity>();

    WHEN("all two hard route constraints are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .hard(InsertionContext{}, view);

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one of all two hard route constraints is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {1}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .hard(InsertionContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two hard route constraints are not fulfilled") {
      THEN("hard returns first code") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {1}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {3}; })
            .hard(InsertionContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("all two soft route constraints returns extra cost") {
      THEN("soft returns their sum") {
        auto result = constraint
            .add([](const auto&) { return 13.1; })
            .add([](const auto&) { return 29.0; })
            .soft(InsertionContext{});

        REQUIRE(result == 42.1);
      }
    }
  }
}

}
