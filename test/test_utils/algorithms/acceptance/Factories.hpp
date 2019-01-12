#pragma once

#include "algorithms/refinement/RefinementContext.hpp"
#include "models/Solution.hpp"
#include "models/common/Cost.hpp"
#include "models/problem/Job.hpp"
#include "utils/Random.hpp"

#include <memory>

namespace vrp::test {

inline algorithms::refinement::RefinementContext
createContext(int generation) {
  return algorithms::refinement::RefinementContext{
    {},
    std::make_shared<utils::Random>(),
    std::make_shared<std::set<models::problem::Job, models::problem::compare_jobs>>(),
    {},
    generation};
}

inline models::EstimatedSolution
createSolution(models::common::Cost cost) {
  return {std::make_shared<models::Solution>(), {cost, 0}};
}

struct select_fake_solution final {
  models::common::Cost cost;
  models::EstimatedSolution operator()(const algorithms::refinement::RefinementContext& ctx) const {
    return createSolution(cost);
  }
};
}