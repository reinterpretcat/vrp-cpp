#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "algorithms/heuristics/Models.hpp"

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

/// Creates evolution context.
template<typename Strategy>
struct create_context final {
  EvolutionContext operator()(const Problem& problem, Strategy& strategy) {
    auto tasks = strategy.population(problem);
    auto population = static_cast<size_t>(tasks.population());
    return {0,
            {static_cast<Problem>(problem), std::move(tasks)},
            vector<Individuum>(population),
            thrust::minstd_rand()};
  }
};

/// Estimates individuum cost.
struct estimate_individuum final {
  calculate_total_cost costs;
  EXEC_UNIT void operator()(Individuum& individuum) { individuum.second = costs(individuum.first); }
};

/// Compares individuums by their cost.
struct compare_individuums final {
  EXEC_UNIT bool operator()(const Individuum& lfs, const Individuum& rhs) {
    return lfs.second < rhs.second;
  }
};

/// Sorts individuums by their cost.
struct sort_individuums final {
  void operator()(EvolutionContext& ctx) {
    auto costs = calculate_total_cost{ctx.solution.getShadow()};
    thrust::for_each(exec_unit, ctx.costs.begin(), ctx.costs.end(), estimate_individuum{costs});
    thrust::sort(exec_unit, ctx.costs.begin(), ctx.costs.end(), compare_individuums{});
  }
};

struct init_individuums final {
  void operator()(EvolutionContext& ctx) {
    thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(ctx.solution.tasks.population()),
                      ctx.costs.begin(), init_individuum{});
    sort_individuums{}(ctx);
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Strategy>
void run_evolution<Strategy>::operator()(const Problem& problem) {
  // TODO pass problem with &&
  auto ctx = create_context<Strategy>{}(problem, strategy);

  init_individuums{}(ctx);

  while (strategy.next(ctx)) {
    // get genetic operators
    auto crossover = strategy.crossover(ctx);
    auto mutator = strategy.mutator(ctx);
    auto selection = strategy.selection(ctx);

    // run selection and apply operators
    select_individuums<decltype(crossover), decltype(mutator)>{crossover, mutator}(ctx, selection);

    sort_individuums{}(ctx);
  }
}

// NOTE explicit specialization to make linker happy.
template class run_evolution<LinearStrategy>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp