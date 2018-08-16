#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Listeners.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "algorithms/genetic/Terminations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"

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

/// Keeps individuum characteristics: index and cost.
using Individuum = thrust::pair<int, float>;

/// Creates individuum.
struct init_individuum final {
  EXEC_UNIT Individuum operator()(int index) { return {index, __FLT_MAX__}; }
};

/// Estimates individuum cost.
struct estimate_individuum final {
  calculate_total_cost costs;
  EXEC_UNIT void operator()(Individuum& individuum) { individuum.second = costs(individuum.first); }
};

/// Sorts individuums by their cost.
struct sort_individuums final {
  calculate_total_cost costs;
  EXEC_UNIT bool operator()(const Individuum& lfs, const Individuum& rhs) {
    return lfs.second < rhs.second;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TerminationCriteria, typename GenerationListener>
void run_evolution<TerminationCriteria, GenerationListener>::operator()(const Problem& problem,
                                                                        const Settings& settings) {
  // data
  auto size = static_cast<size_t>(settings.populationSize);
  auto tasks =
    create_population<nearest_neighbor<TransitionOperator>>{problem}(settings.populationSize);
  auto solution = Solution::Shadow{problem.getShadow(), tasks.getShadow()};
  auto ctx = EvolutionContext{0, {-1, __FLT_MAX__}, vector<Individuum>(size)};

  // operators
  auto costs = calculate_total_cost{solution};
  auto crossover = adjusted_cost_difference<nearest_neighbor<TransitionOperator>>{solution};
  auto mutator = create_mutant<TransitionOperator>{solution};
  auto selection = select_individuums<decltype(crossover), decltype(mutator)>{
    crossover, mutator, thrust::minstd_rand{}};

  // init individuums
  thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(settings.populationSize), ctx.costs.begin(),
                    init_individuum{});

  // TODO change settings based on population diversity
  do {
    // estimate individuums
    thrust::for_each(exec_unit, ctx.costs.begin(), ctx.costs.end(), estimate_individuum{costs});
    thrust::sort(exec_unit, ctx.costs.begin(), ctx.costs.end(), sort_individuums{});

    // selection
    // TODO pass sorted by cost indices
    // TODO define settings
    selection(ctx, {});

    listener(ctx);
    ++ctx.generation;
  } while (!termination(ctx));
}

// NOTE explicit specialization to make linker happy.
template class run_evolution<max_generations, empty_listener>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
