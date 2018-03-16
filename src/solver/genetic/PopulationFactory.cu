#include "algorithms/Costs.cu"
#include "algorithms/Transitions.cu"
#include "models/Transition.hpp"
#include "solver/genetic/PopulationFactory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms;
using namespace vrp::models;
using namespace vrp::genetic;

namespace {

/// Creates roots in solutions. Root connects depot with exact
/// one customer making feasible solution.
struct CreateRoots {
  CreateRoots(const Problem &problem, Tasks &tasks) :
      problem(problem.getShadow()),
      tasks(tasks.getShadow()),
      getCost(problem.resources.getShadow()),
      createTransition(problem.getShadow(), tasks.getShadow()),
      performTransition(tasks.getShadow()) {}

  __host__ __device__
  void operator()(int population) {
    population = population % problem.size;

    int customer = population + 1;
    int task = population * problem.size;

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

  CalculateCost getCost;
  CreateTransition createTransition;
  PerformTransition performTransition;
};

}

namespace vrp {
namespace genetic {

Tasks createPopulation(const Problem &problem,
                       const Settings &settings) {

  Tasks population {problem.size(), settings.populationSize * problem.size()};

  // create roots
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   CreateRoots(problem, population));

  // TODO complete solutions using fast heuristic


  return std::move(population);
}

}
}