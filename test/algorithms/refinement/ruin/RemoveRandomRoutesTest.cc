#include "algorithms/refinement/ruin/RemoveRandomRoutes.hpp"

#include "algorithms/refinement/extensions/RestoreInsertionContext.hpp"
#include "test_utils/algorithms/construction/Results.hpp"
#include "test_utils/algorithms/refinement/MatrixRoutes.hpp"
#include "test_utils/fakes/FakeDistribution.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::models::problem;
using namespace vrp::utils;

namespace vrp::test {

SCENARIO("random routes removal can ruin solution with three routes without locked jobs",
         "[algorithms][refinement][ruin][service]") {
  auto [ints, size] = GENERATE(table<std::vector<int>, size_t>({
    {{2, 0, 2}, 8},
  }));

  GIVEN("solution with 4 routes within 4 service jobs in each") {
    auto [problem, solution] = generate_matrix_routes{}(4, 4);

    WHEN("ruin without locked jobs") {
      auto ctx =
        RefinementContext{problem,
                          std::make_shared<Random>(FakeDistribution<int>{ints, 0}, FakeDistribution<double>{{}, 0}),
                          std::make_shared<std::set<Job, compare_jobs>>(),
                          {},
                          0};
      auto iCtx = restore_insertion_context{}(ctx, *solution);

      remove_random_routes{1, 3, 1}.operator()(ctx, *solution, iCtx);

      THEN("should ruin expected jobs") { REQUIRE(iCtx.solution->required.size() == size); }
    }
  }
}
}
