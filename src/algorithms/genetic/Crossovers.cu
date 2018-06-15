#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/heuristics/Dummy.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::utils;

namespace {

/// Prepare convolution for processing by marking corresponding customer in plan as assigned.
struct prepare_convolution final {
  Solution::Shadow solution;
  size_t base;

  __device__ void operator()(const int task) {
    auto customer = solution.tasks.ids[task];
    // TODO use index of convolution
    solution.tasks.plan[base + customer] = Plan::assign();
  }

  __device__ void operator()(const Convolution& convolution) {
    printf("convolution: [%d, %d]\n", convolution.tasks.first, convolution.tasks.second);
    thrust::for_each(thrust::device, thrust::make_counting_iterator(convolution.tasks.first),
                     thrust::make_counting_iterator(convolution.tasks.second + 1), *this);
  }
};

/// Prepares plan of selected solution which allows "smart" customer reassignment.
struct prepare_plan final {
  Solution::Shadow solution;
  thrust::pair<size_t, thrust::device_ptr<Convolution>> convolutions;

  __device__ void operator()(int index) {
    auto begin = static_cast<size_t>(solution.problem.size) * index;
    auto end = begin + static_cast<size_t>(solution.problem.size);

    printf("offspring: %d, size: %d\n", index, static_cast<int>(convolutions.first));

    // reset whole plan
    thrust::fill(thrust::device, solution.tasks.plan + begin, solution.tasks.plan + end,
                 Plan::empty());

    // mark convolution's customers as assigned
    thrust::for_each(thrust::device, convolutions.second, convolutions.second + convolutions.first,
                     prepare_convolution{solution, begin});
  }
};

/// Creates new solution using convolutions.
struct create_solutions final {
  Solution::Shadow solution;
  thrust::pair<int, int> offspring;
  thrust::pair<size_t, thrust::device_ptr<Convolution>> convolutions;

  __device__ void operator()(int child) {
    auto index = child == 0 ? offspring.first : offspring.second;
    auto begin = solution.problem.size * index;
    auto end = begin + solution.problem.size;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename Heuristic>
__device__ void adjusted_cost_difference<Heuristic>::operator()(
  const Settings& settings,
  const Generation& generation) const {
  // find convolutions
  auto left = create_best_convolutions{solution, pool}.operator()(settings.convolution,
                                                                  generation.parents.first);
  auto right = create_best_convolutions{solution, pool}.operator()(settings.convolution,
                                                                   generation.parents.second);
  auto pairs =
    create_joint_convolutions{solution, pool}.operator()(settings.convolution, left, right);

  auto convolutions =
    create_sliced_convolutions{solution, pool}.operator()(settings.convolution, pairs);

  auto wrapper = thrust::make_pair(convolutions.size, *convolutions.data);

  // prepare offspring
  prepare_plan{solution, wrapper}(generation.offspring.first);
  prepare_plan{solution, wrapper}(generation.offspring.second);

  // reassign customers using convolutions
  thrust::for_each(thrust::device, thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(2),
                   create_solutions{solution, generation.offspring, wrapper});
}

// NOTE explicit specialization to make linker happy.
template class adjusted_cost_difference<vrp::algorithms::heuristics::dummy>;
template class adjusted_cost_difference<vrp::algorithms::heuristics::nearest_neighbor>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
