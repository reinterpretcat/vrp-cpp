#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "utils/random/FilteredDistribution.hpp"

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>

using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

namespace {

/// Generates integer in range [2, max].
inline int generateInt(EvolutionContext& ctx, int max) {
  thrust::uniform_int_distribution<int> dist(2, max);
  return dist(ctx.rng);
}

/// Generates float in range [min, max).
inline float generateFloat(EvolutionContext& ctx, float min, float max) {
  thrust::uniform_real_distribution<float> dist(min, max);
  return dist(ctx.rng);
}

/// Returns elite group size.
inline int getElite(EvolutionContext& ctx) {
  auto max = thrust::max(2, static_cast<int>(ctx.costs.size() * 0.1));
  return generateInt(ctx, max);
}

/// Returns crossover group size.
inline int getCross(EvolutionContext& ctx) {
  auto max = thrust::max(2, static_cast<int>(ctx.costs.size() * 0.2));
  return generateInt(ctx, max);
}

/// Returns mutant group size.
inline int getMutants(EvolutionContext& ctx, int left) {
  assert(left >= 2);
  return left == 2 ? 1 : generateInt(ctx, left / 2);
}

/// Returns convolution settings.
inline vrp::algorithms::convolutions::Settings getConvolutionSettings(EvolutionContext& ctx) {
  auto median = generateFloat(ctx, 0.4, 0.9);
  auto size = generateInt(ctx, 6);
  return {median, size};
}

/// Creates selection settings.
Selection createSelection(EvolutionContext& ctx) {
  auto elite = getElite(ctx);
  auto cross = getCross(ctx);
  auto mutants = getMutants(ctx, static_cast<int>(ctx.costs.size() - elite - cross * 4));

  auto crossSetting = getConvolutionSettings(ctx);
  auto mutantSetting = getConvolutionSettings(ctx);

  assert(elite + cross * 4 + mutants * 2 <= ctx.costs.size());
  return {elite, {cross, crossSetting}, {mutants, mutantSetting}};
}

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

Tasks LinearStrategy::population(const Problem& problem) {
  // NOTE Minimum population: 2 for elite, 2 * 4 for crossover, 2 for mutation.
  assert(settings.populationSize >= 12);

  return create_population<nearest_neighbor<TransitionOperator>>{problem}(settings.populationSize);
}

LinearStrategy::Crossover LinearStrategy::crossover(const EvolutionContext& ctx) {
  return adjusted_cost_difference<nearest_neighbor<TransitionOperator>>{ctx.solution};
}

LinearStrategy::Mutator LinearStrategy::mutator(const EvolutionContext& ctx) {
  return create_mutant<TransitionOperator>{ctx.solution};
}

Selection LinearStrategy::selection(const EvolutionContext& ctx) {
  return createSelection(const_cast<EvolutionContext&>(ctx));
}

bool LinearStrategy::next(EvolutionContext& ctx) {
  std::cout << "generation: " << ctx.generation << " best cost:" << ctx.costs.front().second
            << std::endl;
  ctx.generation++;
  return ctx.generation < 50;
}

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
