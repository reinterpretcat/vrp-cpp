#ifndef VRP_ALGORITHMS_TRANSITIONS_HPP
#define VRP_ALGORITHMS_TRANSITIONS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {

/// Creates transition between customers.
struct create_transition {

  __host__ __device__
  explicit create_transition(const vrp::models::Problem::Shadow &problem,
                             const vrp::models::Tasks::Shadow tasks) :
    problem(problem), tasks(tasks) {}

  __host__ __device__
  vrp::models::Transition operator()(int task, int toCustomer) const {
    int matrix = tasks.ids[task] * problem.size + toCustomer;
    float distance = problem.routing.distances[matrix];
    int traveling = problem.routing.durations[matrix];
    int arrivalTime = tasks.times[task] + traveling;
    int demand = problem.customers.demands[toCustomer];
    int vehicle = tasks.vehicles[task];

    if (isTooLate(toCustomer, arrivalTime) || isTooMuch(task, demand))
      return vrp::models::Transition::createInvalid();

    int waiting = getWaitingTime(toCustomer, arrivalTime);
    int serving = problem.customers.services[toCustomer];
    int departure = arrivalTime + waiting + serving;

    return canReturn(vehicle, toCustomer, departure)
           ? vrp::models::Transition::createInvalid()
           : vrp::models::Transition { toCustomer, vehicle, distance, traveling,
                                       serving, waiting, demand, task + 1 };
  }
 private:
  /// Checks whether vehicle arrives too late.
  __host__ __device__
  inline bool isTooLate(int customer, int arrivalTime) const {
    return arrivalTime > problem.customers.ends[customer];
  }

  /// Checks whether vehicle can carry requested demand.
  __host__ __device__
  inline bool isTooMuch(int task, int demand) const {
    return tasks.capacities[task] < demand;
  }

  /// Calculates waiting time.
  __host__ __device__
  inline int getWaitingTime(int customer, int arrivalTime) const {
    int startTime = problem.customers.starts[customer];
    return arrivalTime < startTime ? startTime - arrivalTime : 0;
  }

  /// Checks whether vehicle can return to depot.
  __host__ __device__
  inline bool canReturn(int vehicle, int toCustomer, int departure) const {
    return departure + problem.routing.durations[toCustomer * problem.size] >
        problem.resources.timeLimits[vehicle];
  }

  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;
};

/// Performs transition with the cost.
struct perform_transition {

  explicit perform_transition(const vrp::models::Tasks::Shadow &tasks) :
    tasks(tasks) {}

  __host__ __device__
  void operator()(const vrp::models::Transition &transition, float cost) const {
    int task = transition.task;

    tasks.ids[task] = transition.customer;
    tasks.times[task] = transition.duration();
    tasks.capacities[task] = tasks.capacities[task - 1] - transition.demand;
    tasks.vehicles[task] = transition.vehicle;

    tasks.costs[task] = cost;
    tasks.plan[base(task) + transition.customer] = true;
  }

 private:
  __host__ __device__
  inline int base(int task) const {
    return (task / tasks.customers) * tasks.customers;
  }

  vrp::models::Tasks::Shadow tasks;
};

}
}

#endif //VRP_TRANSITIONS_HPP
