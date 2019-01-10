#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/solution/Registry.hpp"

#include <numeric>

namespace vrp::algorithms::refinement {

/// Creates refinement context including initial solution from problem.
template<typename Heuristic = construction::CheapestInsertion>
struct create_refinement_context final {
  RefinementContext operator()(const models::Problem& problem) const {
    using namespace vrp::algorithms::construction;

    auto random = std::make_shared<utils::Random>();

    // create initial solution represented by insertion context.
    auto iCtx = Heuristic{InsertionEvaluator{problem.transport, problem.activity}}(
      build_insertion_context{}
        .progress(build_insertion_progress{}.cost(std::numeric_limits<double>::max()).completeness(0).owned())
        .registry(std::make_shared<models::solution::Registry>(*problem.fleet))
        .constraint(problem.constraint)
        .random(random)
        .jobs(problem.jobs->all())
        .owned());

    // create population
    auto population = std::make_shared<std::vector<RefinementContext::Individuum>>();

    return RefinementContext{std::make_shared<models::Problem>(problem), random, {}, population, 0};
  }
};
}
