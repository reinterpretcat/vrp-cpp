#include "Solver.hpp"
#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "algorithms/construction/constraints/VehicleActivityTiming.hpp"
#include "algorithms/objectives/PenalizeUnassignedJobs.hpp"
#include "algorithms/refinement/logging/LogToConsole.hpp"
#include "test_utils/algorithms/construction/Factories.hpp"
#include "test_utils/algorithms/construction/constraints/Helpers.hpp"
#include "test_utils/fakes/TestTransportCosts.hpp"
#include "test_utils/models/Factories.hpp"

#include <catch/catch.hpp>

using namespace vrp::algorithms::refinement;
using namespace vrp::algorithms::objectives;
using namespace vrp::models;
using namespace vrp::models::common;
using namespace vrp::models::costs;
using namespace vrp::models::problem;
using namespace vrp::models::solution;
using namespace vrp::algorithms::construction;
using namespace vrp::test;
using namespace ranges;

namespace {

std::shared_ptr<Fleet>
createFleet() {
  auto fleet = std::make_shared<Fleet>();
  (*fleet).add(test_build_driver{}.owned()).add(test_build_vehicle{}.id("v1").details({{0, {}, {0, 100}}}).owned());
  return fleet;
}


std::shared_ptr<Problem>
createProblem(const std::shared_ptr<Fleet>& fleet, ranges::any_view<Job> jobsView) {
  auto activity = std::make_shared<ActivityCosts>();
  auto transport = std::make_shared<TestTransportCosts>();
  auto objective = std::make_shared<penalize_unassigned_jobs<>>();

  auto constraint = std::make_shared<InsertionConstraint>();
  constraint->template addHard<VehicleActivitySize<int>>(std::make_shared<VehicleActivitySize<int>>());
  constraint->add<VehicleActivityTiming>(std::make_shared<VehicleActivityTiming>(fleet, transport, activity));

  auto jobs = std::make_shared<Jobs>(Jobs{*transport, jobsView, view::single("car")});

  return std::make_shared<Problem>(Problem{fleet, jobs, constraint, objective, activity, transport});
}
}

namespace vrp::test {

SCENARIO("Can solve simple open VRP problem", "[scenarios][openvrp]") {
  auto solver = Solver<create_refinement_context<>,
                       select_best_solution,
                       ruin_and_recreate_solution<>,
                       GreedyAcceptance<>,
                       MaxIterationCriteria,
                       log_to_console>{};

  auto [jobs, size] = GENERATE(std::make_tuple(
                                 std::vector<Job>{
                                   as_job(test_build_service{}.location(5).shared()),
                                   as_job(test_build_service{}.location(10).shared()),
                                 },
                                 3),
                               std::make_tuple(std::vector<Job>{as_job(test_build_service{}.location(5).shared())}, 2));

  GIVEN("An open VRP problem with one or more jobs") {
    auto problem = createProblem(createFleet(), jobs);

    WHEN("run solver") {
      auto estimatedSolution = solver(problem);

      THEN("has proper tour end") {
        REQUIRE(estimatedSolution.first->unassigned.empty());
        REQUIRE(estimatedSolution.first->routes.size() == 1);
        REQUIRE(ranges::size(estimatedSolution.first->routes.front()->tour.activities()) == size);
        REQUIRE(estimatedSolution.first->routes.front()->tour.end()->detail.location != 0);
      }
    }
  }
}
}