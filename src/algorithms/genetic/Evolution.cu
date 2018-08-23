#include "algorithms/costs/SolutionCosts.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <algorithm>
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

/// Compares individuums by their cost.
struct compare_individuums final {
  EXEC_UNIT bool operator()(const Individuum& lfs, const Individuum& rhs) {
    return lfs.second < rhs.second;
  }
};

inline void logCosts(const EvolutionContext& ctx) {
  std::for_each(ctx.costs.begin(), ctx.costs.end(), [](const Individuum& individuum) {
    std::cout << "(" << individuum.first << ", " << individuum.second << ") ";
  });
  std::cout << std::endl << std::endl;
}

/// Sorts individuums by their cost.
struct sort_individuums final {
  void operator()(EvolutionContext& ctx) {
    auto costs = calculate_total_cost{ctx.solution};
    thrust::for_each(exec_unit, ctx.costs.begin(), ctx.costs.end(), estimate_individuum{costs});
    thrust::sort(exec_unit, ctx.costs.begin(), ctx.costs.end(), compare_individuums{});
    logCosts(ctx);
  }
};

struct init_individuums final {
  void operator()(Tasks& tasks, EvolutionContext& ctx) {
    thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(tasks.population()), ctx.costs.begin(),
                      init_individuum{});
    sort_individuums{}(ctx);
  }
};

inline void validateSolution(const EvolutionContext& ctx, const Solution& solution) {
  auto result = vrp::utils::SolutionChecker::check(solution);
  if (!result.isValid()) {
    std::copy(result.errors.begin(), result.errors.end(),
              std::ostream_iterator<std::string>(std::cout, "\n"));
    vrp::streams::MatrixTextWriter::write(std::cout, solution.tasks);
    throw std::runtime_error(std::string("Invalid solution: generation ") +
                             std::to_string(ctx.generation));
  }
}

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Strategy>
void run_evolution<Strategy>::operator()(const Problem& problem) {
  auto solution = Solution{static_cast<Problem>(problem), strategy.population(problem)};
  auto ctx = EvolutionContext{0, solution.getShadow(),
                              vector<Individuum>(static_cast<size_t>(solution.tasks.population())),
                              thrust::minstd_rand()};

  init_individuums{}(solution.tasks, ctx);

  while (strategy.next(ctx)) {
    // get genetic operators
    auto crossover = strategy.crossover(ctx);
    auto mutator = strategy.mutator(ctx);
    auto selection = strategy.selection(ctx);

    // run selection and apply operators
    select_individuums<decltype(crossover), decltype(mutator)>{crossover, mutator}(ctx, selection);

    validateSolution(ctx, solution);

    sort_individuums{}(ctx);
  }
}

// NOTE explicit specialization to make linker happy.
template class run_evolution<LinearStrategy>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp