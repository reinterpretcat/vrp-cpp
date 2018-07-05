#ifndef VRP_HEURISTICS_RANDOMINSERTION_HPP
#define VRP_HEURISTICS_RANDOMINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of random insertion heuristic.
struct random_insertion final {
  const Context& context;

  __host__ __device__ explicit random_insertion(const Context& context) : context(context) {}

  /// Finds the "nearest" transition for given task and vehicle
  __host__ __device__ vrp::models::Transition operator()(const Step& step);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_RANDOMINSERTION_HPP
