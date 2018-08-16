#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "utils/random/FilteredDistribution.hpp"
#include "utils/random/TransformedDistribution.hpp"

#include <thrust/random/normal_distribution.h>
#include <unordered_set>

using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::runtime;
using namespace vrp::utils;

namespace {

/// Crossover selection plan.
struct CrossPlan final {
  thrust::pair<int, int> parents;
  thrust::pair<int, int> children;
};

/// Mutant selection plan.
using MutantPlan = thrust::pair<int, int>;

}  // namespace

namespace std {
template<>
class hash<CrossPlan> {
public:
  size_t operator()(const CrossPlan& plan) const {
    return hash<int>()(plan.parents.first) ^ hash<int>()(plan.parents.second);
  }
};

template<>
class hash<MutantPlan> {
public:
  size_t operator()(const MutantPlan& plan) const { return hash<int>()(plan.first); }
};

/// Do not allow to process items with the same parents.
bool operator==(const CrossPlan& lfs, const CrossPlan& rhs) { return lfs.parents == rhs.parents; }

/// Do not allow to process items with the same original.
bool operator==(const MutantPlan& lfs, const MutantPlan& rhs) { return lfs.first == rhs.first; }

}  // namespace std

namespace {

/// Defines selection data.
struct SelectionData final {
  vector_ptr<thrust::pair<int, float>> costs;
  std::unordered_set<int> candidates;
  std::unordered_set<CrossPlan> cross;
  std::unordered_set<MutantPlan> mutants;
};

/// Finds next index from selection data.
template<typename Distribution>
inline int next(const SelectionData& data,
                Distribution& dist,
                thrust::minstd_rand& rng,
                int other = -1) {
  while (true) {
    auto value = dist(rng);
    if (value != other && data.candidates.find(value) == data.candidates.end()) return value;
  }
}

/// Calls generator with parent and children distributions.
struct with_generator final {
  const EvolutionContext& ctx;
  const Selection& selection;

  template<typename Generator>
  void operator()(const Generator& generator) {
    auto start = 0;
    auto middle = selection.elite - 1;
    auto end = ctx.costs.size() - 1;

    thrust::random::normal_distribution<float> dist(0.0f, 0.5f * end);

    auto tp = transformed(
      dist, [&](float value) -> int { return std::abs(ctx.costs[static_cast<int>(value)].first); });
    auto tc = transformed(dist, [&](float value) -> int {
      return std::abs(ctx.costs[static_cast<int>(end - value)].first);
    });

    auto parents = filtered(tp, [=](int value) { return !(value > end || value < start); });
    auto children = filtered(tc, [=](int value) { return !(value > end || value < middle); });

    generator(parents, children);
  }
};

/// Assigns crossover plan.
struct assign_crossovers final {
  const Selection& settings;
  SelectionData& data;
  thrust::minstd_rand& rng;

  template<typename Parents, typename Children>
  void operator()(Parents& parents, Children& children) const {
    while (data.cross.size() < settings.crossovers.first) {
      auto parent1 = next(data, parents, rng);
      auto parent2 = next(data, parents, rng, parent1);

      if (data.cross.find({{parent1, parent2}, {0, 0}}) != data.cross.end()) continue;

      data.candidates.insert(parent1);
      data.candidates.insert(parent2);

      auto child1 = next(data, children, rng);
      auto child2 = next(data, children, rng, child1);

      data.cross.insert({{parent1, parent2}, {child1, child2}});
      data.candidates.insert(child1);
      data.candidates.insert(child2);
    }
  }
};

/// Assigns crossover plan.
struct assign_mutants final {
  const Selection& settings;
  SelectionData& ctx;
  thrust::minstd_rand& rng;

  template<typename Parents, typename Children>
  void operator()(Parents& parents, Children& children) const {
    while (ctx.mutants.size() < settings.mutations.first) {
      auto parent = next(ctx, parents, rng);

      if (ctx.mutants.find({parent, 0}) != ctx.mutants.end()) continue;

      ctx.candidates.insert(parent);

      auto child = next(ctx, children, rng);

      ctx.mutants.insert({parent, child});
      ctx.candidates.insert(child);
    }
  }
};

/// Runs crossover operator on crossover plan.
template<typename Crossover>
struct apply_crossover final {
  Crossover crossover;
  EXEC_UNIT void operator()(CrossPlan plan) {
    crossover(Generation{plan.parents, plan.children, {}});
  }
};

/// Runs mutator on mutation plan.
template<typename Mutator>
struct apply_mutator final {
  Mutator mutator;
  EXEC_UNIT void operator()(MutantPlan plan) { mutator(Mutation{plan.first, plan.second, {}}); }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Crossover, typename Mutator>
void select_individuums<Crossover, Mutator>::operator()(const EvolutionContext& ctx,
                                                        const Selection& selection) {
  auto data = SelectionData();

  with_generator{ctx, selection}(assign_crossovers{selection, data, rng});
  with_generator{ctx, selection}(assign_mutants{selection, data, rng});

  // TODO pass convolution settings
  thrust::for_each(exec_unit, data.cross.begin(), data.cross.end(),
                   apply_crossover<Crossover>{crossover});
  thrust::for_each(exec_unit, data.mutants.begin(), data.mutants.end(),
                   apply_mutator<Mutator>{mutator});
}

/// NOTE Make linker happy
template class select_individuums<empty_crossover, empty_mutator>;
template class select_individuums<adjusted_cost_difference<nearest_neighbor<TransitionOperator>>,
                                  create_mutant<TransitionOperator>>;
template class select_individuums<adjusted_cost_difference<random_insertion<TransitionOperator>>,
                                  create_mutant<TransitionOperator>>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
