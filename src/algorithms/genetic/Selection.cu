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

/// Defines selection context.
struct SelectionContext final {
  std::unordered_set<CrossPlan> cross;
  std::unordered_set<MutantPlan> mutants;
  std::unordered_set<int> candidates;
};

/// Finds next index from context.
template<typename Distribution>
inline int next(const SelectionContext& context,
                Distribution& dist,
                thrust::minstd_rand& rng,
                int other = -1) {
  while (true) {
    auto value = dist(rng);
    if (value != other && context.candidates.find(value) == context.candidates.end()) return value;
  }
}

/// Calls generator with parent and children distributions.
struct with_generator final {
  const Selection& selection;

  template<typename Generator>
  void operator()(const Generator& generator) {
    auto start = 0;
    auto middle = selection.elite - 1;
    auto end = selection.last;

    thrust::random::normal_distribution<float> dist(0.0f, 0.5f * end);

    auto tp =
      transformed(dist, [](float value) -> int { return std::abs(static_cast<int>(value)); });
    auto tc = transformed(
      dist, [end](float value) -> int { return std::abs(static_cast<int>(end - value)); });

    auto parents = filtered(tp, [=](int value) { return !(value > end || value < start); });
    auto children = filtered(tc, [=](int value) { return !(value > end || value < middle); });

    generator(parents, children);
  }
};

/// Assigns crossover plan.
struct assign_crossovers final {
  const Selection& settings;
  SelectionContext& ctx;
  thrust::minstd_rand& rng;

  template<typename Parents, typename Children>
  void operator()(Parents& parents, Children& children) const {
    while (ctx.cross.size() < settings.crossovers.first) {
      auto parent1 = next(ctx, parents, rng);
      auto parent2 = next(ctx, parents, rng, parent1);

      if (ctx.cross.find({{parent1, parent2}, {0, 0}}) != ctx.cross.end()) continue;

      ctx.candidates.insert(parent1);
      ctx.candidates.insert(parent2);

      auto child1 = next(ctx, children, rng);
      auto child2 = next(ctx, children, rng, child1);

      ctx.cross.insert({{parent1, parent2}, {child1, child2}});
      ctx.candidates.insert(child1);
      ctx.candidates.insert(child2);
    }
  }
};

/// Assigns crossover plan.
struct assign_mutants final {
  const Selection& settings;
  SelectionContext& ctx;
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
void select_individuums<Crossover, Mutator>::operator()(const Selection& selection) {
  auto ctx = SelectionContext();

  with_generator{selection}(assign_crossovers{selection, ctx, rng});
  with_generator{selection}(assign_mutants{selection, ctx, rng});

  // TODO pass convolution settings
  thrust::for_each(exec_unit, ctx.cross.begin(), ctx.cross.end(),
                   apply_crossover<Crossover>{crossover});
  thrust::for_each(exec_unit, ctx.mutants.begin(), ctx.mutants.end(),
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
