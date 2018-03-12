#include "algorithms/Costs.cu"
#include "algorithms/Transitions.cu"
#include "models/Transition.hpp"
#include "solver/genetic/PopulationFactory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

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
    int task = population * (problemSize + 1) + customer;

    while(customer != 0) {
      auto transition = createTransition(startTime, depot, customer, vehicle);
      if (transition.isValid()) {
        performTransition(transition, task, getCost(transition));
        break;
      }
      // NOTE this happens if customer is unreachable by current vehicle.
      // TODO pick another vehicle in case of heterogeneous fleet.
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

/// Intializes solutions by setting all tasks as unprocessed within unique customer.
struct InitPopulation {
  int problemSize;

  __host__ __device__
  void operator()(thrust::tuple<int, int&, int&> tuple) const {
    // get valid customer id
    int i = thrust::get<0>(tuple) % problemSize;
    thrust::get<1>(tuple) = i;
    thrust::get<2>(tuple) = -1;
  }
};

}

namespace vrp {
namespace genetic {

Tasks createPopulation(const Problem &problem,
                       const Settings &settings) {

  Tasks population{settings.populationSize * problem.size()};

  // 1. init population by forming tours, e.g. 0-1-2-3-4-0 without allocating resources.
  thrust::for_each(thrust::device,
                   thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::make_counting_iterator(0),
                       population.ids.begin(),
                       population.vehicles.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::make_counting_iterator(population.size()),
                       population.ids.end(),
                       population.vehicles.end())),
                   InitPopulation {problem.size()});

  // 2. create roots
//  thrust::for_each(thrust::device,
//                   thrust::make_counting_iterator(0),
//                   thrust::make_counting_iterator(settings.populationSize()),
//                   CreateRoots(problem));

  return std::move(population);
}

}
}