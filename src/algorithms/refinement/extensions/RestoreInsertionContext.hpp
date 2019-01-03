#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/problem/Job.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

/// Restores insertion context after solution modifications.
struct restore_insertion_context final {
  construction::InsertionContext operator()(const RefinementContext& ctx, models::Solution& sln) {
    using namespace vrp::algorithms::construction;

    auto routes = std::map<std::shared_ptr<models::solution::Route>, std::shared_ptr<InsertionRouteState>>{};
    ranges::for_each(sln.routes, [&](const auto& route) {
      auto state = std::make_shared<InsertionRouteState>();
      ctx.problem->constraint->accept(*route, *state);
      routes.insert({route, state});
    });

    return construction::build_insertion_context{}
      .progress({sln.cost, static_cast<double>(sln.unassigned.size()) / ctx.problem->jobs->size()})
      .registry(sln.registry)
      .constraint(ctx.problem->constraint)
      .jobs(std::move(sln.unassigned))
      .routes(std::move(routes))
      .random(ctx.random)
      .owned();
  }
};
}