#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Populations.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {
/// Reserves convolution in plan.
struct assign_convolution {
  Solution::Shadow solution;
  const Convolution& convolution;
  size_t base;
  bool preferFirst;

  EXEC_UNIT void operator()(const thrust::tuple<int, int>& tuple) {
    auto task = thrust::get<0>(tuple);
    auto index = thrust::get<1>(tuple);

    int customer = solution.tasks.ids[convolution.base + task];
    Plan current = solution.tasks.plan[base + customer];

    solution.tasks.plan[base + customer] =
      Plan::reserve(current.hasConvolution() && preferFirst ? current.convolution() : index);
  }
};

/// Prepares convolution for processing by marking corresponding customer in plan as assigned.
struct prepare_convolution final {
  Solution::Shadow solution;
  size_t base;
  bool preferFirst;

  EXEC_UNIT void operator()(const thrust::tuple<int, Convolution>& tuple) {
    const auto index = thrust::get<0>(tuple);
    const auto& convolution = thrust::get<1>(tuple);

    thrust::for_each(exec_unit_policy{},
                     thrust::make_zip_iterator(
                       thrust::make_tuple(thrust::make_counting_iterator(convolution.tasks.first),
                                          thrust::make_constant_iterator(index))),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::make_counting_iterator(convolution.tasks.second + 1),
                       thrust::make_constant_iterator(index))),
                     assign_convolution{solution, convolution, base, preferFirst});
  }
};

/// Prepares plan of selected solution which allows "smart" customer reassignment.
struct prepare_plan final {
  Solution::Shadow solution;
  thrust::pair<size_t, vector_ptr<Convolution>> convolutions;
  thrust::pair<int, int> offspring;

  EXEC_UNIT void operator()(int order) {
    auto index = order == 0 ? offspring.first : offspring.second;
    auto begin = static_cast<size_t>(solution.problem.size * index);
    auto end = begin + static_cast<size_t>(solution.problem.size);

    // reset whole plan except depot
    thrust::fill(exec_unit_policy{}, solution.tasks.plan + begin + 1, solution.tasks.plan + end,
                 Plan::empty());

    // mark convolution's customers as assigned
    thrust::for_each(exec_unit_policy{},
                     thrust::make_zip_iterator(
                       thrust::make_tuple(thrust::make_counting_iterator(0), convolutions.second)),
                     thrust::make_zip_iterator(thrust::make_tuple(
                       thrust::make_counting_iterator(static_cast<int>(convolutions.first)),
                       convolutions.second + convolutions.first)),
                     prepare_convolution{solution, begin, order == 0});
  }
};

struct prepare_plans final {
  Solution::Shadow solution;
  thrust::pair<size_t, vector_ptr<Convolution>> convolutions;
  thrust::pair<int, int> offspring;

  EXEC_UNIT void operator()() {
    thrust::for_each(exec_unit_policy{}, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(2),
                     prepare_plan{solution, convolutions, offspring});
  }
};

template<typename Heuristic>
struct improve_individuum final {
  Solution::Shadow solution;
  const vector_ptr<Convolution> convolutions;
  thrust::pair<int, int> offspring;

  EXEC_UNIT void operator()(int order) {
    int index = order == 0 ? offspring.first : offspring.second;
    create_individuum<Heuristic>{solution.problem, solution.tasks, convolutions, 0}.operator()(
      index);
  }
};

template<typename Heuristic>
struct improve_individuums final {
  Solution::Shadow solution;
  const vector_ptr<Convolution> convolutions;
  thrust::pair<int, int> offspring;

  EXEC_UNIT void operator()() {
    // TODO try thrust::seq
    thrust::for_each(exec_unit_policy{}, thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(2),
                     improve_individuum<Heuristic>{solution, convolutions, offspring});
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Heuristic>
EXEC_UNIT void adjusted_cost_difference<Heuristic>::operator()(const Generation& generation) const {
  // find convolutions
  auto left =
    create_best_convolutions{solution}.operator()(generation.settings, generation.parents.first);
  auto right =
    create_best_convolutions{solution}.operator()(generation.settings, generation.parents.second);

  if (left.size == 0 || right.size == 0) return;

  auto pairs = create_joint_convolutions{solution}.operator()(generation.settings, left, right);

  auto convolutions = create_sliced_convolutions{solution}.operator()(generation.settings, pairs);

  auto wrapper = thrust::make_pair(convolutions.size, *convolutions.data);

  prepare_plans{solution, wrapper, generation.offspring}();

  improve_individuums<Heuristic>{solution, *convolutions.data.get(), generation.offspring}();
}

// NOTE explicit specialization to make linker happy.
using TransitionOperator = vrp::algorithms::heuristics::TransitionOperator;
template class adjusted_cost_difference<vrp::algorithms::heuristics::dummy<TransitionOperator>>;
template class adjusted_cost_difference<
  vrp::algorithms::heuristics::nearest_neighbor<TransitionOperator>>;
template class adjusted_cost_difference<
  vrp::algorithms::heuristics::random_insertion<TransitionOperator>>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
