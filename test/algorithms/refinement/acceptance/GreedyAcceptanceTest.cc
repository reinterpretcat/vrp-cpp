#include "algorithms/refinement/acceptance/GreedyAcceptance.hpp"

#include "test_utils/algorithms/acceptance/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models::common;

namespace vrp::test {

SCENARIO("greedy acceptance can work with population", "[algorithms][refinement][acceptance]") {
  auto [bestCost, newCost, expected] = GENERATE(table<Cost, Cost, bool>({
    {1.0, 2.0, false},
    {1.0, 1.0, false},
    {1.0, 0.5, true},
  }));

  GIVEN("refinement context") {
    auto acceptance = GreedyAcceptance<select_fake_solution>{{bestCost}};
    WHEN("accepts greedy") {
      auto result = acceptance(createContext(1), createSolution(newCost));
      THEN("returns expected result") { REQUIRE(result == expected); }
    }
  }
}
}
