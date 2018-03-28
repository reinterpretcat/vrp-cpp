#include "algorithms/Costs.cu"
#include "algorithms/Transitions.cu"
#include "heuristics/NearestNeighbor.hpp"
#include "models/Transition.hpp"
#include "solver/genetic/PopulationFactory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms;
using namespace vrp::heuristics;
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
    int task = individuum * problem.size;

    createDepotTask(task);

    while(customer != 0) {
      auto transition = createTransition(task, customer);
      if (transition.isValid()) {
        performTransition(transition, getCost(transition));
        break;
      }
      // TODO try to pick another vehicle in case of heterogeneous fleet.
      customer = (customer + 1) % problem.size;
    }
  }
 private:

  __host__ __device__
  void createDepotTask(int task) {
    const int vehicle = 0;
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

  calculate_cost getCost;
  create_transition createTransition;
  perform_transition performTransition;
};

/// Completes solutions using fast heuristic (Nearest Neighbor).
struct complete_solution {
  complete_solution(const Problem &problem, Tasks &tasks) :
      problem(problem.getShadow()),
      tasks(tasks.getShadow()),
      performTransition(tasks.getShadow()){}

  __host__ __device__
  void operator()(int individuum) {
    auto heuristic = NearestNeighbor(problem, tasks);
    auto begin = individuum * problem.size + 1;
    auto end = begin + problem.size - 2;
  }

  const Problem::Shadow problem;
  Tasks::Shadow tasks;
  perform_transition performTransition;
};

}

namespace vrp {
namespace genetic {

Tasks createPopulation(const Problem &problem,
                       const Settings &settings) {
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
                   complete_solution(problem, population));

  return std::move(population);
}

}
}