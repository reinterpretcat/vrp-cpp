#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "algorithms/transitions/Executors.hpp"
#include "algorithms/transitions/Factories.hpp"
#include "models/Transition.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

using TransitionOperator = vrp::algorithms::heuristics::TransitionOperator;

/// Creates roots in solutions. Root connects depot with exact
/// one customer making feasible solution.
template<typename TransitionOp>
struct create_roots {
  create_roots(const Problem& problem, Tasks& tasks, int populationSize) :
    problem(problem.getShadow()), tasks(tasks.getShadow()), populationSize(populationSize) {}

  ANY_EXEC_UNIT void operator()(int individuum) {
    auto transitionOp = TransitionOp(problem, tasks);

    int customer = getCustomer(individuum);
    int vehicle = 0;

    int base = individuum * problem.size;
    int fromTask = 0;
    int toTask = fromTask + 1;

    createDepotTask(base + fromTask, vehicle);

    while (customer != 0) {
      auto details =
        vrp::models::Transition::Details{base, fromTask, toTask, wrapCustomer(customer), vehicle};
      auto transition = transitionOp.create(details);
      if (transition.isValid()) {
        auto cost = transitionOp.estimate(transition);
        transitionOp.perform(transition, cost);
        break;
      }
      // TODO try to pick another vehicle in case of heterogeneous fleet.
      customer = (customer + 1) % problem.size;
    }
  }

private:
  ANY_EXEC_UNIT void createDepotTask(int task, int vehicle) {
    const int depot = 0;

    tasks.ids[task] = depot;
    tasks.times[task] = problem.customers.starts[0];
    tasks.capacities[task] = problem.resources.capacities[vehicle];
    tasks.vehicles[task] = vehicle;
    tasks.costs[task] = problem.resources.fixedCosts[vehicle];
    tasks.plan[task] = Plan::assign();
  }

  ANY_EXEC_UNIT inline int getCustomer(int individuum) const {
    return thrust::max(1, (individuum * tasks.customers / populationSize + 1) % tasks.customers);
  }

  ANY_EXEC_UNIT inline variant<int, Convolution> wrapCustomer(int customer) const {
    variant<int, Convolution> wrapped;
    wrapped.set<int>(customer);
    return wrapped;
  };

  const Problem::Shadow problem;
  Tasks::Shadow tasks;
  int populationSize;
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
  thrust::for_each(exec_unit, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(settings.populationSize),
                   create_roots<TransitionOperator>(problem, population, settings.populationSize));

  // complete solutions
  thrust::for_each(
    exec_unit, thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(settings.populationSize),
    create_individuum<Heuristic>{problem.getShadow(), population.getShadow(), {}, 1});

  return std::move(population);
}

template<typename Heuristic>
EXEC_UNIT void create_individuum<Heuristic>::operator()(int index) {
  auto context = Context{problem, tasks, convolutions};

  Heuristic()(context, index, shift);
}

// NOTE explicit specialization to make linker happy.
template class create_population<vrp::algorithms::heuristics::dummy<TransitionOperator>>;
template class create_population<vrp::algorithms::heuristics::nearest_neighbor<TransitionOperator>>;
template class create_population<vrp::algorithms::heuristics::random_insertion<TransitionOperator>>;

template class create_individuum<vrp::algorithms::heuristics::dummy<TransitionOperator>>;
template class create_individuum<vrp::algorithms::heuristics::nearest_neighbor<TransitionOperator>>;
template class create_individuum<vrp::algorithms::heuristics::random_insertion<TransitionOperator>>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
