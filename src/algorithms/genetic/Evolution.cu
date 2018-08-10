#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Listeners.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Terminations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

using namespace thrust::placeholders;

namespace {
/// Keeps individuum characteristics.
struct Individuum final {
  /// Index in population.
  int index;
  /// Total cost.
  float cost;
};

/// Keeps population characteristics.
struct Population final {
  /// Population data.
  Tasks::Shadow tasks;
  /// Keeps all individuums characteristics.
  vector_ptr<Individuum> individuums;
};

/// Creates individuum.
struct init_individuum final {
  EXEC_UNIT Individuum operator()(int index) { return {index, __FLT_MAX__}; }
};

/// Estimates individuum cost.
struct estimate_individuum final {
  calculate_total_cost costs;
  EXEC_UNIT void operator()(Individuum& individuum) { individuum.cost = costs(individuum.index); }
};

/// Sorts individuums by their cost.
struct sort_individuums final {
  calculate_total_cost costs;
  EXEC_UNIT bool operator()(const Individuum& lfs, const Individuum& rhs) {
    return lfs.cost < rhs.cost;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TerminationCriteria, typename GenerationListener>
void run_evolution<TerminationCriteria, GenerationListener>::operator()(const Problem& problem,
                                                                        const Settings& settings) {
  auto tasks =
    create_population<nearest_neighbor<TransitionOperator>>{problem}(settings.populationSize);
  auto context = EvolutionContext{0, __FLT_MAX__, -1};
  auto individuums = vector<Individuum>(static_cast<size_t>(settings.populationSize));
  auto population = Population{tasks.getShadow(), individuums.data()};
  auto costs = calculate_total_cost{Solution::Shadow{problem.getShadow(), tasks.getShadow()}};

  // init population
  thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(settings.populationSize), population.individuums,
                    init_individuum{});

  do {
    // estimate individuums
    thrust::for_each(exec_unit, individuums.begin(), individuums.end(), estimate_individuum{costs});
    thrust::sort(exec_unit, individuums.begin(), individuums.end(), sort_individuums{});

    // Elitism
    // Selection
    // Crossover
    // Mutation

    listener(context);
    ++context.generation;
  } while (!termination(context));
}

// NOTE explicit specialization to make linker happy.
template class run_evolution<max_generations, empty_listener>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
