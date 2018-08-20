#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "utils/random/FilteredDistribution.hpp"
#include "utils/random/TransformedDistribution.hpp"

#include <algorithm>
#include <iostream>
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
                const thrust::minstd_rand& rng,
                int other = -1) {
  while (true) {
    auto value = dist(const_cast<thrust::minstd_rand&>(rng));
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
  const EvolutionContext& ctx;
  const Selection& settings;
  SelectionData& data;

  template<typename Parents, typename Children>
  void operator()(Parents& parents, Children& children) const {
    while (data.cross.size() < settings.crossovers.first) {
      auto parent1 = next(data, parents, ctx.rng);
      auto parent2 = next(data, parents, ctx.rng, parent1);

      if (data.cross.find({{parent1, parent2}, {0, 0}}) != data.cross.end()) continue;

      data.candidates.insert(parent1);
      data.candidates.insert(parent2);

      auto child1 = next(data, children, ctx.rng);
      auto child2 = next(data, children, ctx.rng, child1);

      data.cross.insert({{parent1, parent2}, {child1, child2}});
      data.candidates.insert(child1);
      data.candidates.insert(child2);
    }
  }
};

/// Assigns crossover plan.
struct assign_mutants final {
  const EvolutionContext& ctx;
  const Selection& settings;
  SelectionData& data;

  template<typename Parents, typename Children>
  void operator()(Parents& parents, Children& children) const {
    while (data.mutants.size() < settings.mutations.first) {
      auto parent = next(data, parents, ctx.rng);

      if (data.mutants.find({parent, 0}) != data.mutants.end()) continue;

      data.candidates.insert(parent);

      auto child = next(data, children, ctx.rng);

      data.mutants.insert({parent, child});
      data.candidates.insert(child);
    }
  }
};

/// Runs crossover operator on crossover plan.
template<typename Crossover>
struct apply_crossover final {
  Crossover crossover;
  vrp::algorithms::convolutions::Settings settings;
  EXEC_UNIT void operator()(CrossPlan plan) {
    crossover(Generation{plan.parents, plan.children, settings});
  }
};

/// Runs mutator on mutation plan.
template<typename Mutator>
struct apply_mutator final {
  Mutator mutator;
  vrp::algorithms::convolutions::Settings settings;
  EXEC_UNIT void operator()(MutantPlan plan) {
    mutator(Mutation{plan.first, plan.second, settings});
  }
};

inline void logSelection(const Selection& selection, const SelectionData& data) {
  std::cout << std::endl << std::endl;
  std::cout << "elite: " << selection.elite << std::endl;
  std::cout << "crossovers: " << selection.crossovers.first
            << " median: " << selection.crossovers.second.MedianRatio
            << " size: " << selection.crossovers.second.ConvolutionSize << std::endl;

  std::cout << "cross:\n";
  std::for_each(data.cross.begin(), data.cross.end(), [](const CrossPlan& plan) {
    std::cout << plan.parents.first << " " << plan.parents.second << " " << plan.children.first
              << " " << plan.children.second << std::endl;
  });

  std::cout << "mutants:\n";
  std::for_each(data.mutants.begin(), data.mutants.end(), [](const MutantPlan& plan) {
    std::cout << plan.first << " " << plan.second << std::endl;
  });

  std::cout << "candidates:\n";
  std::copy(data.candidates.begin(), data.candidates.end(),
            std::ostream_iterator<int>(std::cout, ", "));
  std::cout << std::endl;
}

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Crossover, typename Mutator>
void select_individuums<Crossover, Mutator>::operator()(const EvolutionContext& ctx,
                                                        const Selection& selection) {
  auto data = SelectionData();

  with_generator{ctx, selection}(assign_crossovers{ctx, selection, data});
  with_generator{ctx, selection}(assign_mutants{ctx, selection, data});

  logSelection(selection, data);

  thrust::for_each(exec_unit, data.cross.begin(), data.cross.end(),
                   apply_crossover<Crossover>{crossover, selection.crossovers.second});
  thrust::for_each(exec_unit, data.mutants.begin(), data.mutants.end(),
                   apply_mutator<Mutator>{mutator, selection.mutations.second});
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
