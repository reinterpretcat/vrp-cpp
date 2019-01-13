#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/extensions/solution/DeepCopies.hpp"
#include "models/problem/Job.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// Restores insertion context from solution.
/// Please note that empty routes are excluded from result and state is not updated.
struct restore_insertion_context final {
  construction::InsertionContext operator()(const RefinementContext& ctx, const models::Solution& sln) {
    using namespace vrp::algorithms::construction;
    using namespace vrp::models::solution;

    auto registry = deep_copy_registry{}(sln.registry);

    auto jobs = std::set<models::problem::Job, models::problem::compare_jobs>{};
    ranges::for_each(sln.unassigned, [&](const auto& j) { jobs.insert(j.first); });

    auto routes = std::map<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>>{};
    ranges::for_each(sln.routes, [&](const auto& r) {
      if (r->tour.empty())
        registry->free(*r->actor);
      else
        routes.insert({deep_copy_route{}(r), std::make_shared<InsertionRouteState>()});
    });

    return construction::build_insertion_context{}
      .progress(build_insertion_progress{}
                  .cost(std::numeric_limits<double>::max())
                  .completeness(1 - static_cast<double>(sln.unassigned.size()) / ctx.problem->jobs->size())
                  .total(ctx.problem->jobs->size())
                  .owned())
      .registry(registry)
      .constraint(ctx.problem->constraint)
      .jobs(std::move(jobs))
      .routes(std::move(routes))
      .random(ctx.random)
      .owned();
  }
};
}