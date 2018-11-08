#include "algorithms/constraints/CompositeConstraint.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::constraints;

namespace vrp::test {

SCENARIO("composite constraint can handle multiple constraints", "[algorithms][constraints]") {
  GIVEN("composite constraint with two hard route constraints") {
    auto composite = CompositeConstraint{};

    WHEN("both fulfilled") {
      THEN("fulfilled returns true") {
        auto result = composite.fulfilled();
        // TODO
      }
    }

    WHEN("one is not fulfilled") {
      THEN("fulfilled returns false") {
        auto result = composite.fulfilled();
        // TODO
      }
    }
  }
}

}
