#include "algorithms/heuristics/RandomInsertion.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

using namespace vrp::algorithms::heuristics;
using namespace vrp::models;

namespace {}

namespace vrp {
namespace algorithms {
namespace heuristics {

Transition random_insertion::operator()(const Step& step) {
  //  return thrust::transform_reduce(
  //      thrust::device,
  //      thrust::make_zip_iterator(
  //          thrust::make_tuple(thrust::make_counting_iterator(0), tasks.plan + base)),
  //      thrust::make_zip_iterator(thrust::make_tuple(
  //          thrust::make_counting_iterator(problem.size), tasks.plan + base + problem.size)),
  //      create_cost_transition{base, fromTask, toTask, vehicle, convolutions,
  //                             create_transition{problem, tasks},
  //                             calculate_transition_cost{problem.resources}},
  //      createInvalid(), compare_transition_costs())
  //      .first;
}

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
