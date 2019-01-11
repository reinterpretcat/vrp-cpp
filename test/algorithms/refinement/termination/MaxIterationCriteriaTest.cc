#include "algorithms/refinement/termination/MaxIterationCriteria.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;

namespace {

inline RefinementContext
createContext(int generation) {
  return RefinementContext{
    {}, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>(), {}, generation};
}
}

namespace vrp::test {

SCENARIO("max iteration criteria", "[algorithms][refinement][termination]") {
  auto [max, iteration, expected] = GENERATE(table<int, int, bool>({
    {10, 9, false},
    {10, 10, false},
    {10, 11, true},
  }));

  GIVEN("refinement context") {
    auto termination = MaxIterationCriteria{max};
    WHEN("termination criteria") {
      auto result = termination(createContext(iteration), EstimatedSolution{}, false);
      THEN("returns expected result") { REQUIRE(result == expected); }
    }
  }
}
}
