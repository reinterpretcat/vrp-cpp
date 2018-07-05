#ifndef VRP_HEURISTICS_RANDOMINSERTION_HPP
#define VRP_HEURISTICS_RANDOMINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"
#include "models/Convolution.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "models/Transition.hpp"

#include <thrust/device_ptr.h>

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of random insertion heuristic.
struct random_insertion final {
  const vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
  const thrust::device_ptr<vrp::models::Convolution> convolutions;

  __host__ __device__
  random_insertion(const vrp::models::Problem::Shadow problem,
                   const vrp::models::Tasks::Shadow tasks,
                   const thrust::device_ptr<vrp::models::Convolution> convolutions) :
    problem(problem),
    tasks(tasks), convolutions(convolutions) {}

  /// Finds the "nearest" transition for given task and vehicle
  __host__ __device__ vrp::models::Transition operator()(const Step& step);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_RANDOMINSERTION_HPP
