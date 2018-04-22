#include "algorithms/Costs.cu"
#include "algorithms/Transitions.cu"
#include "models/Transition.hpp"
#include "solver/genetic/Populations.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::genetic;

namespace {

/// Creates roots in solutions. Root connects depot with exact
/// one customer making feasible solution.
struct create_roots {
  create_roots(const Problem &problem, Tasks &tasks) :
      problem(problem.getShadow()),
      tasks(tasks.getShadow()),
      getCost(problem.resources.getShadow()),
      createTransition(problem.getShadow(), tasks.getShadow()),
      performTransition(tasks.getShadow()) {}

  __host__ __device__
  void operator()(int individuum) {
    int customer = individuum + 1;
    int vehicle = 0;

    int fromTask = individuum * problem.size;
    int toTask = fromTask + 1;

    createDepotTask(fromTask, vehicle);

    while(customer != 0) {
      auto details = vrp::models::Transition::Details { fromTask, toTask, customer, vehicle };
      auto transition = createTransition(details);
      if (transition.isValid()) {
        performTransition({transition, getCost(transition) });
        break;
      }
      // TODO try to pick another vehicle in case of heterogeneous fleet.
      customer = (customer + 1) % problem.size;
    }
  }
 private:

  __host__ __device__
  void createDepotTask(int task, int vehicle) {
    const int depot = 0;

    tasks.ids[task] = depot;
    tasks.times[task] = problem.customers.starts[0];
    tasks.capacities[task] = problem.resources.capacities[vehicle];
    tasks.vehicles[task] = vehicle;
    tasks.costs[task] = problem.resources.fixedCosts[vehicle];
    tasks.plan[task] = true;
  }

  const Problem::Shadow problem;
  Tasks::Shadow tasks;

  calculate_transition_cost getCost;
  create_transition createTransition;
  perform_transition performTransition;
};

/// Completes solutions using heuristic specified.
template <typename Heuristic>
struct complete_solution {
  complete_solution(const Problem &problem, Tasks &tasks) :
      problem(problem.getShadow()),
      tasks(tasks.getShadow()),
      performTransition(tasks.getShadow()){}

  __host__ __device__
  void operator()(int individuum) {
    const auto begin = individuum * problem.size;
    const auto end = begin + problem.size;

    auto heuristic = Heuristic(problem, tasks);
    int vehicle = 0;
    int from = begin + 1;
    int to = from + 1;

    do {
      auto transitionCost = heuristic(from, to, vehicle);
      if (thrust::get<0>(transitionCost).isValid()) {
        performTransition(transitionCost);
        from = to++;
      } else {
        // NOTE cannot find any further customer to serve within vehicle
        if (from == begin || vehicle == problem.resources.vehicles - 1) break;

        from = begin;
        spawnNewVehicle(from, ++vehicle);
      }

    } while(to < end);
  }

 private:
  /// Picks the next vehicle and assigns it to the task.
  __host__ __device__
  void spawnNewVehicle(int task, int vehicle) {
    tasks.times[task] = problem.customers.starts[0];
    tasks.capacities[task] = problem.resources.capacities[vehicle];
    tasks.costs[task] = problem.resources.fixedCosts[vehicle];
  }

  const Problem::Shadow problem;
  Tasks::Shadow tasks;
  perform_transition performTransition;
};

}

namespace vrp {
namespace genetic {

template<typename Heuristic>
Tasks create_population<Heuristic>::operator()(const Settings &settings) {
  if (settings.populationSize > problem.size()) {
    throw std::invalid_argument("Population size is bigger than problem size.");
  }

  Tasks population {problem.size(), settings.populationSize * problem.size()};

  // create roots
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   create_roots(problem, population));

  // complete solutions
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   complete_solution<Heuristic>(problem, population));

  return std::move(population);
}

// NOTE explicit specialization to make linker happy.
template class create_population<vrp::heuristics::no_transition>;
template class create_population<vrp::heuristics::nearest_neighbor>;

}
}