#include "Strategies.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Strategies.hpp"

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

namespace vrp {
namespace algorithms {
namespace genetic {

Tasks LinearStrategy::population(const Problem& problem) {
  return create_population<nearest_neighbor<TransitionOperator>>{problem}(settings.populationSize);
}

LinearStrategy::Crossover LinearStrategy::crossover(const EvolutionContext& ctx) {
  return adjusted_cost_difference<nearest_neighbor<TransitionOperator>>{ctx.solution};
}

LinearStrategy::Mutator LinearStrategy::mutator(const EvolutionContext& ctx) {
  return create_mutant<TransitionOperator>{ctx.solution};
}

Selection LinearStrategy::selection(const EvolutionContext& ctx) {
  // TODO change settings based on population diversity
  return Selection();
}

bool LinearStrategy::next(EvolutionContext& ctx) {
  ctx.generation++;
  return false;
}

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
