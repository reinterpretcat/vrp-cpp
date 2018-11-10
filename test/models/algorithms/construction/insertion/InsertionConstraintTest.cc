#include "algorithms/construction/insertion/InsertionConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][constraints]") {
  GIVEN("insertion constraint with two hard constraints") {
    auto constraint = InsertionConstraint{};
    auto view = ranges::view::empty<const models::solution::Activity>();

    WHEN("both are fulfilled") {
      THEN("hard returns no value") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .hard(InsertionContext{}, view);

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {1}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {}; })
            .hard(InsertionContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }

    WHEN("both not fulfilled") {
      THEN("hard returns first code") {
        auto result = constraint
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {1}; })
            .add([](const auto&, const auto&) { return InsertionConstraint::HardResult {3}; })
            .hard(InsertionContext{}, view);

        REQUIRE(result.value() == 1);
      }
    }
  }
}

}
