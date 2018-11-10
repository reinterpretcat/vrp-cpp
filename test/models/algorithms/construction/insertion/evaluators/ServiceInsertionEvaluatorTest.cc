#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"

#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][constraints]") {
  GIVEN("service with location") {
    auto constraint = std::make_shared<InsertionConstraint>();
    auto evaluator = ServiceInsertionEvaluator{constraint};
    auto service = DefaultService;

    WHEN("context") {
      THEN("TODO") {

      }
    }

  }
}

}