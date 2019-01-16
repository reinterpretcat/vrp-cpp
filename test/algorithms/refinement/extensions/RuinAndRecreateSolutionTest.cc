#include "algorithms/refinement/RuinAndRecreateSolution.hpp"

#include "algorithms/refinement/extensions/CreateRefinementContext.hpp"
#include "streams/in/Solomon.hpp"
#include "test_utils/streams/SolomonStreams.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::streams::in;

namespace vrp::test {

SCENARIO("ruin and recreate can solve c101 problem", "[solver][default]") {
  auto rar = ruin_and_recreate_solution<>{};

  GIVEN("C101 problem with 25 customers") {
    auto stream = create_c101_25_problem_stream{}();
    auto problem = read_solomon_type<cartesian_distance<1>>{}.operator()(stream);
    auto ctx = create_refinement_context<>{}(problem);

    WHEN("ruin and recreate") {
      auto estimatedSolution = rar(ctx, ctx.population->front());

      THEN("has solution with all jobs assigned") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(!estimatedSolution.first->routes.empty());
      }
    }
  }
}
}
