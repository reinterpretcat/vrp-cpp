#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Listeners.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Terminations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

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
  /// Rank in population.
  int rank;
};

/// Keeps population characteristics.
struct Population final {
  /// Population data.
  Tasks::Shadow tasks;
  /// Keeps all individuums characteristics.
  vector_ptr<Individuum> individuums;
};

/// Creates individuum.
struct create_individuum final {
  EXEC_UNIT Individuum operator()(int index) { return {index, __FLT_MAX__, 0}; }
};

/// Initializes population.
struct init_population {
  Population population;
  ANY_EXEC_UNIT void operator()(int populationSize) {
    thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(populationSize), population.individuums,
                      create_individuum{});
  }
};

/// Calculates total cost for each individuum and ranks them.
struct rank_population final {
  Population population;

  ANY_EXEC_UNIT void operator()() {}
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

  init_population{population}(settings.populationSize);

  do {
    // TODO
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
