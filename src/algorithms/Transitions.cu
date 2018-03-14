#ifndef VRP_ALGORITHMS_TRANSITIONS_HPP
#define VRP_ALGORITHMS_TRANSITIONS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

namespace vrp {
namespace algorithms {

/// Creates transition between customers.
struct CreateTransition {
  const vrp::models::Problem::Shadow problem;

  explicit CreateTransition(const vrp::models::Problem::Shadow &problem) :
    problem(problem) {}

  /// @param time         Current time.
  /// @param vehicle      id of vehicle performs transition.
  /// @param fromCustomer Customer from which vehicle is moving.
  /// @param toCustomer   Customer to which vehicle is moving.
  __host__ __device__
  vrp::models::Transition operator()(int time, int vehicle, int fromCustomer, int toCustomer) const {
    int matrix = fromCustomer * problem.size + toCustomer;
    float distance = problem.routing.distances[matrix];
    int traveling = problem.routing.durations[matrix];
    int arrivalTime = time + traveling;
    int demand = problem.customers.demands[toCustomer];

    if (isTooLate(toCustomer, arrivalTime) || isTooMuch(vehicle, demand))
      return vrp::models::Transition::createInvalid();

    int waiting = getWaitingTime(toCustomer, arrivalTime);
    int serving = problem.customers.services[toCustomer];
    int departure = arrivalTime + waiting + serving;

    return canReturn(vehicle, toCustomer, departure)
           ? vrp::models::Transition::createInvalid()
           : vrp::models::Transition { toCustomer, vehicle, distance, traveling, serving, waiting, demand };
  }
 private:

  /// Checks whether vehicle arrives too late.
  __host__ __device__
  inline bool isTooLate(int customer, int arrivalTime) const {
    return arrivalTime > problem.customers.ends[customer];
  }

  /// Checks whether vehicle can carry requested demand.
  __host__ __device__
  inline bool isTooMuch(int vehicle, int demand) const {
    return problem.resources.capacities[vehicle] < demand;
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
};

/// Performs transition.
struct PerformTransition {
  vrp::models::Tasks::Shadow tasks;

  explicit PerformTransition(const vrp::models::Tasks::Shadow &tasks) :
    tasks(tasks) {}

  __host__ __device__
  void operator()(const vrp::models::Transition &transition, int task, float cost) const {
    tasks.ids[task] = transition.customer;
    tasks.costs[task] = cost;
    tasks.times[task] = transition.duration();
    tasks.vehicles[task] = transition.vehicle;
    // TODO determine proper index
    tasks.plan[transition.customer] = true;
  }
};

}
}

#endif //VRP_TRANSITIONS_HPP
