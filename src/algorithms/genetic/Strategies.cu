#include "algorithms/genetic/Populations.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "streams/output/MatrixTextWriter.hpp"
#include "utils/random/FilteredDistribution.hpp"
#include "utils/validation/SolutionChecker.hpp"

#include <algorithm>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>

using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::utils;

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

inline void logContext(const EvolutionContext& ctx) {
  // general
  std::cout << "generation: " << ctx.generation
            << " best cost:" << static_cast<thrust::pair<int, float>>(ctx.costs.front()).second
            << std::endl;
  // all costs
  std::for_each(ctx.costs.begin(), ctx.costs.end(), [](thrust::pair<int, float> individuum) {
    std::cout << "(" << individuum.first << ", " << individuum.second << ") ";
  });
  std::cout << std::endl << std::endl;
}

inline void validateSolution(const EvolutionContext& ctx) {
  auto result = SolutionChecker::check(ctx.solution);
  if (!result.isValid()) {
    std::copy(result.errors.begin(), result.errors.end(),
              std::ostream_iterator<std::string>(std::cout, "\n"));
    vrp::streams::MatrixTextWriter::write(std::cout, ctx.solution.tasks);
    throw std::runtime_error(std::string("Invalid solution: generation ") +
                             std::to_string(ctx.generation));
  }
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

LinearStrategy::Crossover LinearStrategy::crossover(EvolutionContext& ctx) {
  return Crossover{ctx.solution.getShadow()};
}

LinearStrategy::Mutator LinearStrategy::mutator(EvolutionContext& ctx) {
  return Mutator{ctx.solution.getShadow()};
}

Selection LinearStrategy::selection(EvolutionContext& ctx) { return createSelection(ctx); }

bool LinearStrategy::next(EvolutionContext& ctx) {
  logContext(ctx);
  validateSolution(ctx);

  ctx.generation++;
  return ctx.generation < 50;
}

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
