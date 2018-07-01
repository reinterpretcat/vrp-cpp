#include "algorithms/transitions/Factories.hpp"

using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

/// Gets id of the customer.
__host__ __device__ inline int getCustomerId(const device_variant<int, Convolution>& customer) {
  return customer.is<Convolution>() ? customer.get<Convolution>().customers.first
                                    : customer.get<int>();
}

/// Checks whether vehicle arrives too late.
__host__ __device__ inline bool isTooLate(const Problem::Shadow& problem,
                                          const device_variant<int, Convolution>& customer,
                                          int arrivalTime) {
  int endTime = customer.is<Convolution>() ? customer.get<Convolution>().times.second
                                           : problem.customers.ends[customer.get<int>()];
  return arrivalTime > endTime;
}

/// Checks whether vehicle can carry requested demand.
__host__ __device__ inline bool isTooMuch(const Tasks::Shadow& tasks, int task, int demand) {
  return tasks.capacities[task] < demand;
}

/// Returns demand of transition.
__host__ __device__ inline int getDemand(const Problem::Shadow& problem,
                                         const device_variant<int, Convolution>& customer) {
  return customer.is<Convolution>() ? customer.get<Convolution>().demand
                                    : problem.customers.demands[customer.get<int>()];
}

/// Calculates waiting time.
__host__ __device__ inline int getWaitingTime(const Problem::Shadow& problem,
                                              const device_variant<int, Convolution>& customer,
                                              int arrivalTime) {
  int startTime = customer.is<Convolution>() ? customer.get<Convolution>().times.first
                                             : problem.customers.starts[customer.get<int>()];
  return arrivalTime < startTime ? startTime - arrivalTime : 0;
}

/// Calculates service time.
__host__ __device__ inline int getServiceTime(const Problem::Shadow& problem,
                                              const device_variant<int, Convolution>& customer) {
  return customer.is<Convolution>() ? customer.get<Convolution>().service
                                    : problem.customers.services[customer.get<int>()];
}

/// Checks whether vehicle can NOT return to depot.
__host__ __device__ inline bool noReturn(const Problem::Shadow& problem,
                                         const device_variant<int, Convolution>& customer,
                                         int vehicle,
                                         int departure) {
  int index =
    customer.is<Convolution>() ? customer.get<Convolution>().customers.second : customer.get<int>();

  auto returnTime = departure + problem.routing.durations[index * problem.size];
  return returnTime > problem.resources.timeLimits[vehicle] ||
         returnTime > problem.customers.ends[0];
}

}  // namespace

__host__ __device__ Transition
create_transition::operator()(const Transition::Details& details) const {
  int task = details.base + details.from;
  int customer = getCustomerId(details.customer);

  int matrix = tasks.ids[task] * problem.size + customer;
  float distance = problem.routing.distances[matrix];
  int traveling = problem.routing.durations[matrix];
  int arrivalTime = tasks.times[task] + traveling;
  int demand = getDemand(problem, details.customer);

  if (isTooLate(problem, details.customer, arrivalTime) || isTooMuch(tasks, task, demand)) {
    return vrp::models::Transition();
  }

  int waiting = getWaitingTime(problem, details.customer, arrivalTime);
  int serving = getServiceTime(problem, details.customer);
  int departure = arrivalTime + waiting + serving;

  return noReturn(problem, details.customer, details.vehicle, departure)
           ? Transition()
           : Transition(details, {distance, traveling, serving, waiting, demand});
}
