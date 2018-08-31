#include "algorithms/common/Convolutions.hpp"
#include "algorithms/common/Tours.hpp"
#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "iterators/Aggregates.hpp"
#include "iterators/Sequential.hpp"
#include "models/Plan.hpp"
#include "runtime/Config.hpp"

using namespace vrp::algorithms::common;
using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::algorithms::heuristics;
using namespace vrp::iterators;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Reserves subtour in individuum.
struct reserve_subtour final {
  Tasks::Shadow tasks;
  int base;

  EXEC_UNIT void operator()(thrust::tuple<const Convolution&, int> tuple) {
    auto convolution = thrust::get<0>(tuple);
    auto index = thrust::get<1>(tuple);
    // TODO use parallel foreach?
    for_seq(thrust::make_counting_iterator(convolution.tasks.first),
            thrust::make_counting_iterator(convolution.tasks.second + 1), [&](int, int task) {
              auto customer = tasks.ids[convolution.base + task];
              if (customer != 0) tasks.plan[base + customer] = Plan::reserve(index);
            });
  }
};

/// Reserves tour in individuum if it good enough.
struct reserve_good_tour final {
  Solution::Shadow solution;
  Mutation mutation;
  vector_ptr<Convolution> convolutions;
  int srcBase;
  int dstBase;

  EXEC_UNIT void operator()(const thrust::tuple<int, int, int, int>& range) {
    if (thrust::get<0>(range) == -1) return;

    auto first = srcBase + thrust::get<0>(range);
    auto last = srcBase + thrust::get<1>(range);
    auto index = thrust::get<2>(range);

    // filter by remaining capacity ratio
    auto remain = getRemainCapacityRatio(index, last);
    if (remain > mutation.settings.MedianRatio) return;

    auto convolution = create_convolution{solution}(srcBase, first, last);
    reserve_subtour{solution.tasks, dstBase}({convolution, index});
    convolutions[index] = convolution;
  }

  EXEC_UNIT float getRemainCapacityRatio(int index, int task) {
    float total = solution.problem.resources.capacities[index];
    return (total - solution.tasks.capacities[task]) / total;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TransitionOp>
EXEC_UNIT void mutate_weak_subtours<TransitionOp>::operator()(const Mutation& mutation) const {
  // NOTE RA implementation cannot work in place with convolutions
  assert(mutation.source != mutation.destination);

  int base = solution.problem.size * mutation.destination;

  auto convolutions =
    create_best_convolutions{solution}.operator()(mutation.settings, mutation.source);
  auto convPtr = *convolutions.data.get();

  // reset plan according to convolutions
  thrust::fill(exec_unit_policy{}, base + solution.tasks.plan + 1,
               base + solution.tasks.plan + solution.problem.size, Plan::empty());
  thrust::for_each_n(
    exec_unit_policy{},
    thrust::make_zip_iterator(thrust::make_tuple(convPtr, thrust::make_counting_iterator(0))),
    convolutions.size, reserve_subtour{solution.tasks, base});

  // run RA
  auto context = Context{solution.problem, solution.tasks, convPtr};
  random_insertion<TransitionOperator>{}(context, mutation.destination, 0);
}

template<typename TransitionOp>
EXEC_UNIT void mutate_weak_tours<TransitionOp>::operator()(const Mutation& mutation) const {
  assert(mutation.source != mutation.destination);

  int srcBase = solution.problem.size * mutation.source;
  int dstBase = solution.problem.size * mutation.destination;
  int lastVehicle = solution.tasks.vehicles[srcBase - 1];
  auto convolutions = make_unique_ptr_data<Convolution>(static_cast<size_t>(lastVehicle + 1));

  // reset plan
  thrust::fill(exec_unit_policy{}, dstBase + solution.tasks.plan + 1,
               dstBase + solution.tasks.plan + solution.problem.size, Plan::empty());

  // find tours
  auto iterator = vrp::iterators::make_aggregate_output_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(0), thrust::make_constant_iterator(0),
                         solution.tasks.vehicles, thrust::make_constant_iterator(0))),
    reserve_good_tour{solution, mutation, *convolutions.get(), srcBase, dstBase});

  find_tours<decltype(solution.tasks.vehicles), decltype(iterator)>{lastVehicle}(
    solution.tasks.vehicles + srcBase, solution.tasks.vehicles + srcBase + solution.problem.size,
    iterator);

  // run RA
  auto context = Context{solution.problem, solution.tasks, *convolutions.get()};
  random_insertion<TransitionOperator>{}(context, mutation.destination, 0);
}

/// NOTE make linker happy.
template class mutate_weak_subtours<TransitionOperator>;
template class mutate_weak_tours<TransitionOperator>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
