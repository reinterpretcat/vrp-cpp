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

/// Completes solutions using heuristic specified.
template <typename Heuristic>
struct complete_solution {
  complete_solution(const Problem &problem, Tasks &tasks) :
      problem(problem.getShadow()),
      tasks(tasks.getShadow()),
      performTransition(tasks.getShadow()){}

  __host__ __device__
  void operator()(int individuum) {
    auto heuristic = Heuristic(problem, tasks);
    auto begin = individuum * problem.size + 1;
    auto end = begin + problem.size - 2;

    do {
      auto transitionCost = heuristic(begin);
      if (transitionCost.first.isValid()) {
        performTransition(transitionCost.first, transitionCost.second);
        ++begin;
      } else {
        // TODO use a new vehicle or exit
        break;
      }
    } while(begin < end);
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
template class create_population<vrp::heuristics::NoTransition>;
template class create_population<vrp::heuristics::NearestNeighbor>;

}
}