#include "algorithms/transitions/Executors.hpp"

using namespace vrp::algorithms::transitions;
using namespace vrp::models;

namespace {

__host__ __device__
inline int base(const Tasks::Shadow &tasks, int task) {
  return (task / tasks.customers) * tasks.customers;
}

}

__host__ __device__
void perform_transition::operator()(const TransitionCost &transitionCost) const {
  const auto &transition = thrust::get<0>(transitionCost);
  const auto &details = transition.details;
  const auto &delta = transition.delta;

  tasks.ids[details.to] = details.customer;
  tasks.times[details.to] = tasks.times[details.from] + delta.duration();
  tasks.capacities[details.to] = tasks.capacities[details.from] - delta.demand;
  tasks.vehicles[details.to] = details.vehicle;

  tasks.costs[details.to] = tasks.costs[details.from] + thrust::get<1>(transitionCost);
  tasks.plan[base(tasks, details.to) + details.customer] = true;
}