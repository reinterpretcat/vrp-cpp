#pragma once

#include "Solver.hpp"
#include "test_utils/algorithms/refinement/LogAndValidate.hpp"

namespace vrp::test {

template<typename Stream, typename Reader>
struct solve_stream final {
  std::pair<std::shared_ptr<const models::Problem>, std::shared_ptr<const models::Solution>> operator()() const {
    auto solver = Solver<algorithms::refinement::create_refinement_context<>,
                         algorithms::refinement::select_best_solution,
                         algorithms::refinement::ruin_and_recreate_solution<>,
                         algorithms::refinement::GreedyAcceptance<>,
                         algorithms::refinement::MaxIterationCriteria,
                         vrp::test::log_and_validate>{};

    auto stream = Stream{}();

    auto problem = Reader{}.operator()(stream);
    auto estimatedSolution = solver(problem);

    return {problem, estimatedSolution.first};
  }
};
}