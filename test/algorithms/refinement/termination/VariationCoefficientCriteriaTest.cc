#include "algorithms/refinement/termination/VariationCoefficientCriteria.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models;
using namespace vrp::models::common;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::utils;
using namespace ranges;

namespace {

inline RefinementContext
createContext(int generation) {
  return RefinementContext{
    {}, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>(), {}, generation};
}

inline EstimatedSolution
getSolutionWithCost(double cost) {
  return std::make_pair<std::shared_ptr<const Solution>, ObjectiveCost>({}, ObjectiveCost{cost, 0});
}
}

namespace vrp::test {

SCENARIO("variation coefficient criteria", "[algorithms][refinement][termination]") {
  auto [iterations, threshold, delta, expected] = GENERATE(table<size_t, double, double, std::vector<char>>({
    {5, 0.1, 1E-2, std::vector<char>{'f', 'f', 'f', 'f', 't'}},
    {5, 0.1, 1E-1, std::vector<char>{'f', 'f', 'f', 'f', 'f'}},
  }));

  GIVEN("variation coefficient termination") {
    auto termination = VariationCoefficientCriteria{iterations, threshold};
    WHEN("multiple solution are accepted") {
      auto result =
        ranges::view::transform(
          view::closed_indices(1, static_cast<int>(iterations)) | view::reverse,
          [iterations = iterations, delta = delta, &termination](auto i) {
            return termination(createContext(iterations - i + 1), getSolutionWithCost(1 + i * delta), true) ? 't' : 'f';
          }) |
        to_vector;

      THEN("returns expected result") { CHECK_THAT(result, Catch::Equals(expected)); }
    }
  }
}
}
