#include "algorithms/costs/TransitionCosts.hpp"
#include "algorithms/heuristics/NearestNeighbor.hpp"
#include "algorithms/heuristics/Operators.hpp"
#include "algorithms/transitions/Factories.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

using namespace vrp::algorithms::costs;
using namespace vrp::algorithms::heuristics;
using namespace vrp::algorithms::transitions;
using namespace vrp::models;
using namespace vrp::utils;

Transition nearest_neighbor::operator()(const Step& step) {
  TransitionCostOp operators = {create_transition{context.problem, context.tasks},
                                calculate_transition_cost{context.problem, context.tasks}};

  return thrust::transform_reduce(
           thrust::device,
           thrust::make_zip_iterator(
             thrust::make_tuple(thrust::make_counting_iterator(0), context.tasks.plan + step.base)),
           thrust::make_zip_iterator(
             thrust::make_tuple(thrust::make_counting_iterator(context.problem.size),
                                context.tasks.plan + step.base + context.problem.size)),
           create_cost_transition(step, operators, context.convolutions), create_invaild(),
           compare_transition_costs())
    .first;
}
