#pragma once

#include "algorithms/construction/extensions/Factories.hpp"
#include "algorithms/construction/heuristics/CheapestInsertion.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Problem.hpp"
#include "models/solution/Registry.hpp"

#include <numeric>

namespace vrp::algorithms::refinement {

/// Creates refinement context including initial solution from problem.
template<typename Heuristic = construction::CheapestInsertion>
struct create_refinement_context final {
  RefinementContext operator()(const std::shared_ptr<const models::Problem>& problem) const {
    using namespace ranges;
    using namespace vrp::algorithms::construction;
    using Population = RefinementContext::Population;
    using LockedJobs = std::set<models::problem::Job, models::problem::compare_jobs>;

    // TODO remove seed for production use
    auto random = std::make_shared<utils::Random>(0);

    // create initial solution represented by insertion context.
    auto iCtx =
      Heuristic{InsertionEvaluator{}}(build_insertion_context{}
                                        .progress(build_insertion_progress{}
                                                    .cost(models::common::NoCost)
                                                    .completeness(0)
                                                    .total(static_cast<int>(problem->jobs->size()))
                                                    .owned())
                                        .registry(std::make_shared<models::solution::Registry>(*problem->fleet))
                                        .problem(problem)
                                        .random(random)
                                        .jobs(problem->jobs->all())
                                        .owned());

    // create solution and calculate its cost
    auto sln = std::make_shared<models::Solution>(
      models::Solution{iCtx.registry,
                       iCtx.routes | view::transform([](const auto& rs) {
                         return static_cast<std::shared_ptr<const models::solution::Route>>(rs.route);
                       }) |
                         to_vector,
                       std::move(iCtx.unassigned)});
    auto cost = problem->objective->operator()(*sln, *problem->activity, *problem->transport);

    return RefinementContext{problem,
                             random,
                             std::make_shared<const LockedJobs>(),
                             std::make_shared<Population>(Population{models::EstimatedSolution{sln, cost}}),
                             0};
  }
};
}
