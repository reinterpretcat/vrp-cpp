#ifndef VRP_UTILS_TASKUTILS_HPP
#define VRP_UTILS_TASKUTILS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"


namespace vrp {
namespace test {

inline void createDepotTask(const vrp::models::Problem& problem, vrp::models::Tasks& tasks) {
  const int vehicle = 0;
  const int depot = 0;
  const int task = 0;

  tasks.ids[task] = depot;
  tasks.times[task] = problem.customers.starts[0];
  tasks.capacities[task] = problem.resources.capacities[vehicle];
  tasks.vehicles[task] = vehicle;
  tasks.costs[task] = problem.resources.fixedCosts[vehicle];
  tasks.plan[task] = true;
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_TASKUTILS_HPP
