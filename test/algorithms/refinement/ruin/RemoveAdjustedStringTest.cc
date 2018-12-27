#include "algorithms/refinement/ruin/RemoveAdjustedString.hpp"

#include "test_utils/algorithms/refinement/MatrixRoutes.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models::problem;
using namespace vrp::utils;

namespace {

template<typename T>
struct FakeDistribution {
  T value;
  T operator()(T min, T max) { return value; }
};
}

namespace vrp::test {

SCENARIO("adjusted string removal can ruin solution", "[algorithms][refinement][ruin]") {
  auto ras = RemoveAdjustedString{};

  GIVEN("solution with 4 routes within 5 service jobs in each") {
    auto [problem, solution] = generate_matrix_routes{}(4, 5);

    WHEN("ruin without locked jobs") {
      auto context = RefinementContext{{}, std::make_shared<Random>(), std::make_shared<std::set<Job, compare_jobs>>()};

      THEN("should ruin expected jobs") {
        // TODO
      }
    }
  }
}
}
