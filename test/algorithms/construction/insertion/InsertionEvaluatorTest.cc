#include "algorithms/construction/insertion/InsertionEvaluator.hpp"

#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::models::costs;

namespace vrp::test {

SCENARIO("insertion evaluator can handle service insertion", "[algorithms][construction][insertion]") {
  GIVEN("insertion evaluator and service") {
    auto fleet = std::make_shared<Fleet>();
    (*fleet)  //
      .add(test_build_driver{}.owned())
      .add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned())
      .add(test_build_vehicle{}.id("v2").details({{0, {}, {0, 100}}, {1, {}, {0, 50}}}).owned());

    auto constraint = std::make_shared<InsertionConstraint>();
    auto evaluator =
      InsertionEvaluator{fleet, std::make_shared<TestTransportCosts>(), std::make_shared<ActivityCosts>(), constraint};

    WHEN("TODO") {
      THEN("TODO") {}
    }
  }
}
}
