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
struct NoTransition {
  __host__ __device__
  NoTransition(const vrp::models::Problem::Shadow problem,
        vrp::models::Tasks::Shadow tasks) {}

  __host__ __device__
  thrust::pair<vrp::models::Transition, float> operator()(int task) {
    return thrust::make_pair(vrp::models::Transition(), -1);
  };
};

}
}

#endif //VRP_HEURISTICS_NOTRANSITION_HPP