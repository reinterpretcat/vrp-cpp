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
      problemSize(problem.size()),
      startTime(problem.customers.starts[0]),
      getCost(problem.resources.getShadow()),
      createTransition(problem.getShadow()),
      performTransition(tasks.getShadow()) {}

  __host__ __device__
  void operator()(int population) const {
    population = population % problemSize;

    int vehicle = 0;
    int depot = 0;
    int customer = population + 1;
    int task = population * problemSize + 1;

    while(customer != 0) {
      auto transition = createTransition(startTime, vehicle, depot, customer);
      if (transition.isValid()) {
        performTransition(transition, task, getCost(transition));
        break;
      }
      // TODO try to pick another vehicle in case of heterogeneous fleet.
      customer = (customer + 1) % problemSize;
    }
  }
 private:
  int problemSize;
  int startTime;

  CalculateCost getCost;
  CreateTransition createTransition;
  PerformTransition performTransition;
};

}

namespace vrp {
namespace genetic {

Tasks createPopulation(const Problem &problem,
                       const Settings &settings) {

  Tasks population {settings.populationSize * problem.size()};

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