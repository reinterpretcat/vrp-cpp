#include "algorithms/refinement/acceptance/ThresholdAcceptance.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::common;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;

namespace {

inline RefinementContext
createContext(int generation) {
  return RefinementContext{
    {}, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>(), {}, generation};
}

inline RefinementContext::Individuum
createSolution(Cost cost) {
  return std::make_shared<std::pair<Cost, Solution>>(cost, Solution{});
}

struct select_fake_solution final {
  Cost cost;
  RefinementContext::Individuum operator()(const RefinementContext& ctx) const { return createSolution(cost); }
};
}

namespace vrp::test {

SCENARIO("threshold acceptance can work with population", "[algorithms][refinement][acceptance]") {
  auto [initial, bestCost, newCost, generation, expected] = GENERATE(table<Cost, Cost, Cost, int, bool>({
    {0.0, 1.0, 1.100, 1, false},
    {0.0, 1.0, 0.900, 1, true},
    {0.0, 1.0, 1.000, 1, true},
    {0.5, 1.0, 1.100, 1, true},
    {0.5, 1.0, 1.498, 1, true},
    {0.5, 1.0, 1.499, 1, false},  // 0.5 * exp(-log(2)* (1/1000) / 0.3) = 0.498846...
    {0.5, 1.0, 1.498, 2, false},  // 0.5 * exp(-log(2)* (2/1000) / 0.3) = 0.497695...
  }));

  GIVEN("refinement context") {
    auto acceptance = ThresholdAcceptance<select_fake_solution>{initial, 0.3, 1000, {bestCost}};
    WHEN("accepts with threshold") {
      auto result = acceptance(createContext(generation), createSolution(newCost));
      THEN("returns expected result") { REQUIRE(result == expected); }
    }
  }
}
}
