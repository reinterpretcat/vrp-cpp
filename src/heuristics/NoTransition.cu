#ifndef VRP_HEURISTICS_NOTRANSITION_HPP
#define VRP_HEURISTICS_NOTRANSITION_HPP

#include "models/Tasks.hpp"
#include "models/Transition.hpp"
#include "models/Problem.hpp"

#include <thrust/execution_policy.h>

namespace vrp {
namespace heuristics {

/// A dummy implementation of heuristic which returns
/// an invalid transition and cost.
struct no_transition final {
  __host__ __device__
  no_transition(const vrp::models::Problem::Shadow problem,
               vrp::models::Tasks::Shadow tasks) {}

  __host__ __device__
  vrp::models::TransitionCost operator()(int fromTask, int toTask, int vehicle) {
    return thrust::make_tuple(vrp::models::Transition(), -1);
  };
};

}
}

#endif //VRP_HEURISTICS_NOTRANSITION_HPP