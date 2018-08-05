#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/heuristics/Models.hpp"
#include "algorithms/heuristics/RandomInsertion.hpp"
#include "iterators/Sequential.hpp"
#include "models/Plan.hpp"
#include "runtime/Config.hpp"

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::heuristics;
using namespace vrp::iterators;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

struct reserve_convolution final {
  Tasks::Shadow tasks;
  int base;

  EXEC_UNIT void operator()(thrust::tuple<const Convolution&, int> tuple) {
    auto convolution = thrust::get<0>(tuple);
    auto index = thrust::get<1>(tuple);
    for_seq(thrust::make_counting_iterator(convolution.tasks.first),
            thrust::make_counting_iterator(convolution.tasks.second + 1),
            [&](int, int task) { tasks.plan[base + task] = Plan::reserve(index); });
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TransitionOp>
EXEC_UNIT void create_mutant<TransitionOp>::operator()(const Mutation& mutation) const {
  // NOTE RA implementation cannot work in place with convolutions
  assert(mutation.source != mutation.destination);

  int base = solution.problem.size * mutation.destination;

  auto settings = vrp::algorithms::convolutions::Settings{0.75, 0.05};
  auto convolutions = create_best_convolutions{solution}.operator()(settings, mutation.source);
  auto convPtr = *convolutions.data.get();

  // reset plan according to convolutions
  thrust::fill(exec_unit_policy{}, base + solution.tasks.plan,
               base + solution.tasks.plan + convolutions.size, Plan::empty());
  thrust::for_each_n(
    exec_unit_policy{},
    thrust::make_zip_iterator(thrust::make_tuple(convPtr, thrust::make_counting_iterator(0))),
    convolutions.size, reserve_convolution{solution.tasks, base});

  // run RA
  auto context = Context{solution.problem, solution.tasks, convPtr};
  random_insertion<TransitionOperator>{}(context, mutation.destination, 0);
}

/// NOTE make linker happy.
template class create_mutant<TransitionOperator>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
