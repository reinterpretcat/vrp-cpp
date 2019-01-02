#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/problem/Job.hpp"

#include <range/v3/all.hpp>

namespace vrp::algorithms::refinement {

struct RecreateWithBlinks final {
  void operator()(const RefinementContext& ctx, models::Solution& sln) const {
    using namespace ranges;

    //    auto insertion = construction::build_insertion_context{}
    //      .constraint(ctx.problem->constraint)
    //      .jobs(ctx.problem->)
    //      .owned()
  }
};
}