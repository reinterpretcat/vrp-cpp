#pragma once

#include "algorithms/construction/extensions/Factories.hpp"
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
    auto jobs = sln.unassigned | ranges::view::transform([&](const auto& j) { return j.first; }) | ranges::to_vector;

    auto routes = std::set<InsertionRouteContext, compare_insertion_route_contexts>{};
    ranges::for_each(sln.routes, [&](const auto& r) {
      if (r->tour.hasJobs()) {
        auto context = InsertionRouteContext{deep_copy_route{}(r), std::make_shared<InsertionRouteState>()};
        routes.insert(context);
        ctx.problem->constraint->accept(context);
      } else {
        registry->free(r->actor);
      }
    });

    return construction::build_insertion_context{}
      .progress(build_insertion_progress{}
                  .cost(std::numeric_limits<models::common::Cost>::max())
                  .completeness(1 - static_cast<double>(sln.unassigned.size()) / ctx.problem->jobs->size())
                  .total(static_cast<int>(ctx.problem->jobs->size()))
                  .owned())
      .registry(registry)
      .problem(ctx.problem)
      .jobs(std::move(jobs))
      .routes(std::move(routes))
      .random(ctx.random)
      .owned();
  }
};
}