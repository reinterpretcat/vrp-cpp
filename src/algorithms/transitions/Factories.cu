#include "algorithms/transitions/Factories.hpp"

using namespace vrp::algorithms::transitions;
using namespace vrp::models;

namespace {

/// Checks whether vehicle arrives too late.
__host__ __device__ inline bool isTooLate(const Problem::Shadow& problem,
                                          const Transition::Details& details,
                                          int arrivalTime) {
  auto customer = details.customer.get<int>();
  return arrivalTime > problem.customers.ends[customer];
}

/// Checks whether vehicle can carry requested demand.
__host__ __device__ inline bool isTooMuch(const Tasks::Shadow& tasks, int task, int demand) {
  return tasks.capacities[task] < demand;
}

/// Calculates waiting time.
__host__ __device__ inline int getWaitingTime(const Problem::Shadow& problem,
                                              const Transition::Details& details,
                                              int arrivalTime) {
  auto customer = details.customer.get<int>();
  int startTime = problem.customers.starts[customer];
  return arrivalTime < startTime ? startTime - arrivalTime : 0;
}

/// Checks whether vehicle can NOT return to depot.
__host__ __device__ inline bool noReturn(const Problem::Shadow& problem,
                                         const Transition::Details& details,
                                         int departure) {
  auto customer = details.customer.get<int>();
  return departure + problem.routing.durations[customer * problem.size] >
         problem.resources.timeLimits[details.vehicle];
}

}  // namespace

__host__ __device__ Transition
create_transition::operator()(const Transition::Details& details) const {
  int task = details.from;
  int customer = details.customer.get<int>();

  int matrix = tasks.ids[task] * problem.size + customer;
  float distance = problem.routing.distances[matrix];
  int traveling = problem.routing.durations[matrix];
  int arrivalTime = tasks.times[task] + traveling;
  int demand = problem.customers.demands[customer];

  if (isTooLate(problem, details, arrivalTime) || isTooMuch(tasks, task, demand)) {
    return vrp::models::Transition();
  }

  int waiting = getWaitingTime(problem, details, arrivalTime);
  int serving = problem.customers.services[customer];
  int departure = arrivalTime + waiting + serving;

  return noReturn(problem, details, departure)
           ? Transition()
           : Transition(details, {distance, traveling, serving, waiting, demand});
}
