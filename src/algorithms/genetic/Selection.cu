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
#include <vector>

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
/// Individuum index with its cost.
using Individuum = thrust::pair<int, float>;

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

struct Allocation {
  int start;
  int middle;
  int end;
};

/// Copies data on host to vector
/// NOTE mostly needed because unordered set does not provide random access iterator.
template<typename T>
vector<T> copy(const std::unordered_set<T>& source) {
  auto data = vector<T>(source.size());
  std::copy(source.begin(), source.end(), data.begin());
  return std::move(data);
}

/// Defines selection data.
struct SelectionData final {
  SelectionData(const EvolutionContext& ctx, const Selection& selection) :
    alloc{0, selection.elite - 1, static_cast<int>(ctx.costs.size() - 1)},
    costs(const_cast<EvolutionContext&>(ctx).costs.data()), candidates(), cross(), mutants() {}

  Allocation alloc;
  vector_ptr<Individuum> costs;
  std::unordered_set<int> candidates;
  std::unordered_set<CrossPlan> cross;
  std::unordered_set<MutantPlan> mutants;
};

inline bool isNext(const SelectionData& data, int value, int other = -1) {
  return value != other && data.candidates.find(value) == data.candidates.end();
}

/// Finds next index from selection data.
template<typename Distribution>
inline int next(const SelectionData& data,
                Distribution& dist,
                const thrust::minstd_rand& rng,
                int other = -1) {
  auto value = dist(const_cast<thrust::minstd_rand&>(rng));

  if (isNext(data, value, other)) return value;

  // search next with decreasing priority
  for (int i = value; i <= data.alloc.end; ++i)
    if (isNext(data, i, other)) return i;

  // search next with increasing priority
  for (int i = value; i > data.alloc.middle; --i)
    if (isNext(data, i, other)) return i;

  throw std::runtime_error("Cannot get next index.");
}

/// Calls generator with parent and children distributions.
struct with_generator final {
  const SelectionData& data;

  template<typename Generator>
  void operator()(const Generator& generator) {
    auto start = data.alloc.start;
    auto middle = data.alloc.middle;
    auto end = data.alloc.end;

    thrust::random::normal_distribution<float> dist(0.0f, 0.5f * end);

    auto parents = filtered(dist, [=](int value) { return value >= start && value <= end; });
    auto children = filtered(dist, [=](int value) { return value > middle && value <= end; });

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
  vector_ptr<thrust::pair<int, float>> index;

  EXEC_UNIT void operator()(CrossPlan plan) {
    Individuum parent1 = index[plan.parents.first];
    Individuum parent2 = index[plan.parents.second];

    /// NOTE quick test for similarity to avoid crossover call
    if (parent1.second == parent2.second) return;

    Individuum child1 = index[plan.children.first];
    Individuum child2 = index[plan.children.second];

    thrust::pair<int, int> parents = {parent1.first, parent2.first};
    thrust::pair<int, int> children = {child1.first, child2.first};
    crossover(Generation{parents, children, settings});
  }
};

/// Runs mutator on mutation plan.
template<typename Mutator>
struct apply_mutator final {
  Mutator mutator;
  vrp::algorithms::convolutions::Settings settings;
  vector_ptr<thrust::pair<int, float>> index;

  EXEC_UNIT void operator()(MutantPlan plan) {
    Individuum source = index[plan.first];
    Individuum destination = index[plan.second];
    mutator(Mutation{source.first, destination.first, settings});
  }
};

inline void logSelection(const Selection& selection, const SelectionData& data) {
  std::cout << std::endl << std::endl;
  std::cout << "elite: " << selection.elite << std::endl;
  std::cout << "crossovers: " << selection.crossovers.first
            << " median: " << selection.crossovers.second.MedianRatio
            << " size: " << selection.crossovers.second.ConvolutionSize << std::endl;

  std::cout << "cross:\n";
  std::for_each(data.cross.begin(), data.cross.end(), [&](const CrossPlan& plan) {
    std::cout << static_cast<Individuum>(data.costs[plan.parents.first]).first << " "
              << static_cast<Individuum>(data.costs[plan.parents.second]).first << " "
              << static_cast<Individuum>(data.costs[plan.children.first]).first << " "
              << static_cast<Individuum>(data.costs[plan.children.second]).first << std::endl;
  });

  std::cout << "mutants:\n";
  std::for_each(data.mutants.begin(), data.mutants.end(), [&](const MutantPlan& plan) {
    std::cout << static_cast<Individuum>(data.costs[plan.first]).first << " "
              << static_cast<Individuum>(data.costs[plan.second]).first << std::endl;
  });

  std::cout << std::endl;
}

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Crossover, typename Mutator>
void select_individuums<Crossover, Mutator>::operator()(const EvolutionContext& ctx,
                                                        const Selection& selection) {
  auto data = SelectionData(ctx, selection);

  with_generator{data}(assign_crossovers{ctx, selection, data});
  with_generator{data}(assign_mutants{ctx, selection, data});

  logSelection(selection, data);

  auto crossPlan = copy(data.cross);
  thrust::for_each(exec_unit, crossPlan.begin(), crossPlan.end(),
                   apply_crossover<Crossover>{crossover, selection.crossovers.second, data.costs});
  auto mutantPlan = copy(data.mutants);
  thrust::for_each(exec_unit, mutantPlan.begin(), mutantPlan.end(),
                   apply_mutator<Mutator>{mutator, selection.mutations.second, data.costs});
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
