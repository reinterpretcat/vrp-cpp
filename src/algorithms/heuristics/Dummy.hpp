#ifndef VRP_HEURISTICS_DUMMY_HPP
#define VRP_HEURISTICS_DUMMY_HPP

#include "algorithms/heuristics/Models.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// A dummy implementation of heuristic which returns an invalid transition and cost.
struct dummy final {
  ANY_EXEC_UNIT dummy(const vrp::models::Problem::Shadow problem,
                      const vrp::models::Tasks::Shadow tasks,
                      const vrp::runtime::vector_ptr<vrp::models::Convolution> convolutions) {}

  ANY_EXEC_UNIT vrp::models::Transition operator()(const Step& step);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_DUMMY_HPP
