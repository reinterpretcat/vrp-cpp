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
  vrp::models::Transition operator()(const vrp::models::TransitionTask &transition) const {
    const auto &details = thrust::get<0>(transition);
    const int task = thrust::get<1>(transition);

    int matrix = tasks.ids[task] * problem.size + details.customer;
    float distance = problem.routing.distances[matrix];
    int traveling = problem.routing.durations[matrix];
    int arrivalTime = tasks.times[task] + traveling;
    int demand = problem.customers.demands[details.customer];

    if (isTooLate(details, arrivalTime) || isTooMuch(task, demand)) {
      return vrp::models::Transition();
    }

    int waiting = getWaitingTime(details, arrivalTime);
    int serving = problem.customers.services[details.customer];
    int departure = arrivalTime + waiting + serving;

    return noReturn(details, departure)
           ? vrp::models::Transition()
           : vrp::models::Transition(details, {distance, traveling, serving, waiting, demand});
  }
 private:
  /// Checks whether vehicle arrives too late.
  __host__ __device__
  inline bool isTooLate(const vrp::models::Transition::Details &details, int arrivalTime) const {
    return arrivalTime > problem.customers.ends[details.customer];
  }

  /// Checks whether vehicle can carry requested demand.
  __host__ __device__
  inline bool isTooMuch(int task, int demand) const {
    return tasks.capacities[task] < demand;
  }

  /// Calculates waiting time.
  __host__ __device__
  inline int getWaitingTime(const vrp::models::Transition::Details &details, int arrivalTime) const {
    int startTime = problem.customers.starts[details.customer];
    return arrivalTime < startTime ? startTime - arrivalTime : 0;
  }

  /// Checks whether vehicle can NOT return to depot.
  __host__ __device__
  inline bool noReturn(const vrp::models::Transition::Details &details, int departure) const {
    return departure + problem.routing.durations[details.customer * problem.size] >
        problem.resources.timeLimits[details.vehicle];
  }

  const vrp::models::Problem::Shadow problem;
  const vrp::models::Tasks::Shadow tasks;
};

/// Performs transition with a cost.
struct perform_transition {

  explicit perform_transition(const vrp::models::Tasks::Shadow &tasks) :
    tasks(tasks) {}

  __host__ __device__
  void operator()(const vrp::models::TransitionComplete &complete) const {
    const auto fromTask = thrust::get<1>(complete);
    const auto toTask = thrust::get<2>(complete);

    const auto &transition = thrust::get<0>(complete);
    const auto &details = thrust::get<0>(transition).details;
    const auto &delta = thrust::get<0>(transition).delta;
    const auto cost = thrust::get<1>(transition);

    tasks.ids[toTask] = details.customer;
    tasks.times[toTask] = delta.duration();
    tasks.capacities[toTask] = tasks.capacities[fromTask] - delta.demand;
    tasks.vehicles[toTask] = details.vehicle;

    tasks.costs[toTask] = cost;
    tasks.plan[base(toTask) + details.customer] = true;
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
