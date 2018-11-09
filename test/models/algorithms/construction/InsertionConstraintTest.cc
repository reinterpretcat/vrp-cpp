#include "algorithms/construction/insertion/InsertionConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("insertion constraint can handle multiple constraints", "[algorithms][constraints]") {
  GIVEN("composite constraint with two hard constraints") {
    auto constraint = InsertionConstraint{};

    WHEN("both are fulfilled") {
      THEN("hard no value") {
        auto result = constraint
            .add([]() { return InsertionConstraint::HardResult {}; })
            .add([]() { return InsertionConstraint::HardResult {}; })
            .hard();

        REQUIRE(!result.has_value());
      }
    }

    WHEN("one is fulfilled") {
      THEN("hard returns single code") {
        auto result = constraint
            .add([]() { return InsertionConstraint::HardResult {1}; })
            .add([]() { return InsertionConstraint::HardResult {}; })
            .hard();

        CHECK_THAT(result.value(),  Catch::Matchers::Equals(std::vector<int>{1}));
      }
    }

    WHEN("both not fulfilled") {
      THEN("hard returns both codes") {
        auto result = constraint
            .add([]() { return InsertionConstraint::HardResult {1}; })
            .add([]() { return InsertionConstraint::HardResult {3}; })
            .hard();

        CHECK_THAT(result.value(),  Catch::Matchers::Equals(std::vector<int>{1, 3}));
      }
    }
  }
}

}
