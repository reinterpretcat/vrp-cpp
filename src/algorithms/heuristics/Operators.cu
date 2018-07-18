#include "algorithms/heuristics/Operators.hpp"

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Returns cost without service time.
__host__ __device__ inline float getCost(const Transition& transition,
                                         calculate_transition_cost& costFunc) {
  auto newTransition = transition;
  newTransition.delta.serving = 0;
  return costFunc(transition);
}

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

__host__ __device__ TransitionCostModel
create_cost_transition::operator()(const thrust::tuple<int, Plan>& customer) {
  auto plan = thrust::get<1>(customer);

  if (plan.isAssigned()) return create_invaild();

  auto wrapped = device_variant<int, Convolution>();
  if (plan.hasConvolution()) {
    wrapped.set<Convolution>(*(convolutions + plan.convolution()));
  } else
    wrapped.set<int>(thrust::get<0>(customer));

  auto transition = operators.first({step.base, step.from, step.to, wrapped, step.vehicle});

  float cost = transition.isValid() ? getCost(transition, operators.second) : -1;

  return thrust::make_pair(transition, cost);
}

TransitionCostModel& compare_transition_costs::operator()(TransitionCostModel& result,
                                                          const TransitionCostModel& left) {
  if (left.first.isValid() && (left.second < result.second || !result.first.isValid())) {
    result = left;
  }

  return result;
}

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp