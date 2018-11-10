#include "algorithms/construction/insertion/evaluators/ServiceInsertionEvaluator.hpp"

#include "test_utils/algorithms/construction/Insertions.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::construction;

namespace vrp::test {

SCENARIO("service insertion evaluator", "[algorithms][constraints]") {
  GIVEN("insertable service with location") {
    auto constraint = std::make_shared<InsertionConstraint>();
    auto evaluator = ServiceInsertionEvaluator{constraint};
    auto route = test_build_route{}.owned();

    WHEN("evaluate insertion context with empty tour") {
      auto result = evaluator.evaluate(DefaultService,
                                       test_build_insertion_context{}.owned(), 1000);

      THEN("returns insertion success") {
        // TODO
      }
    }

  }
}

}