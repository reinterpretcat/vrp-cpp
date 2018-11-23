#include "algorithms/construction/insertion/InsertionEvaluator.hpp"

#include "test_utils/fakes/TestTransportCosts.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;

namespace vrp::test {

SCENARIO("insertion evaluator can handle service insertion", "[algorithms][construction][insertion]") {
  GIVEN("insertion evaluator and service") {
    auto constraint = std::make_shared<InsertionConstraint>();
    auto evaluator =
      InsertionEvaluator{std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>(), constraint};

    WHEN("TODO") {
      THEN("TODO") {}
    }
  }
}
}
