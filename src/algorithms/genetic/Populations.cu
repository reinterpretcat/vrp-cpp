#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "models/Transition.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

/// Creates roots in solutions. Root connects depot with exact
/// one customer making feasible solution.
struct create_roots {
  create_roots(const Problem& problem, Tasks& tasks, const Settings& settings) :
    problem(problem.getShadow()), tasks(tasks.getShadow()), populationSize(settings.populationSize),
    getCost(problem.resources.getShadow()),
    createTransition(problem.getShadow(), tasks.getShadow()),
    performTransition(problem.getShadow(), tasks.getShadow()) {}

  __host__ __device__ void operator()(int individuum) {
    int customer = getCustomer(individuum);
    int vehicle = 0;

    int fromTask = individuum * problem.size;
    int toTask = fromTask + 1;

    createDepotTask(fromTask, vehicle);

    while (customer != 0) {
      auto details =
        vrp::models::Transition::Details{fromTask, toTask, wrapCustomer(customer), vehicle};
      auto transition = createTransition(details);
      if (transition.isValid()) {
        performTransition(transition, getCost(transition));
        break;
      }
      // TODO try to pick another vehicle in case of heterogeneous fleet.
      customer = (customer + 1) % problem.size;
    }
  }

private:
  __host__ __device__ void createDepotTask(int task, int vehicle) {
    const int depot = 0;

    tasks.ids[task] = depot;
    tasks.times[task] = problem.customers.starts[0];
    tasks.capacities[task] = problem.resources.capacities[vehicle];
    tasks.vehicles[task] = vehicle;
    tasks.costs[task] = problem.resources.fixedCosts[vehicle];
    tasks.plan[task] = Plan::assign();
  }

  __host__ __device__ inline int getCustomer(int individuum) const {
    return thrust::max(1, (individuum * tasks.customers / populationSize + 1) % tasks.customers);
  }

  __host__ __device__ inline device_variant<int, Convolution> wrapCustomer(int customer) const {
    device_variant<int, Convolution> wrapped;
    wrapped.set<int>(customer);
    return wrapped;
  };

  const Problem::Shadow problem;
  Tasks::Shadow tasks;
  int populationSize;

  calculate_transition_cost getCost;
  create_transition createTransition;
  perform_transition performTransition;
};

/// Completes solutions using heuristic specified.
template<typename Heuristic>
struct complete_solution {
  complete_solution(const Problem& problem, Tasks& tasks) :
    problem(problem.getShadow()), tasks(tasks.getShadow()), getCost(problem.resources.getShadow()),
    performTransition(problem.getShadow(), tasks.getShadow()) {}

  __host__ __device__ void operator()(int individuum) {
    const auto begin = individuum * problem.size;
    const auto end = begin + problem.size;

    auto heuristic = Heuristic(problem, tasks);
    int vehicle = 0;
    int from = begin + 1;
    int to = from + 1;

    do {
      auto transition = heuristic(from, to, vehicle);
      if (transition.isValid()) {
        performTransition(transition, getCost(transition));
        from = to++;
      } else {
        // NOTE cannot find any further customer to serve within vehicle
        if (from == begin || vehicle == problem.resources.vehicles - 1) break;

        from = begin;
        spawnNewVehicle(from, ++vehicle);
      }

    } while (to < end);
  }

private:
  /// Picks the next vehicle and assigns it to the task.
  __host__ __device__ void spawnNewVehicle(int task, int vehicle) {
    tasks.times[task] = problem.customers.starts[0];
    tasks.capacities[task] = problem.resources.capacities[vehicle];
    tasks.costs[task] = problem.resources.fixedCosts[vehicle];
  }

  const Problem::Shadow problem;
  Tasks::Shadow tasks;
  calculate_transition_cost getCost;
  perform_transition performTransition;
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Heuristic>
Tasks create_population<Heuristic>::operator()(const Settings& settings) {
  if (settings.populationSize > problem.size()) {
    throw std::invalid_argument("Population size is bigger than problem size.");
  }

  Tasks population(problem.size(), settings.populationSize * problem.size());

  // create roots
  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   create_roots(problem, population, settings));

  // complete solutions
  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   complete_solution<Heuristic>(problem, population));

  return std::move(population);
}

// NOTE explicit specialization to make linker happy.
template class create_population<vrp::algorithms::heuristics::dummy>;
template class create_population<vrp::algorithms::heuristics::nearest_neighbor>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp