#ifndef VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
#define VRP_HEURISTICS_NEARESTNEIGHBOR_HPP

#include "algorithms/heuristics/Models.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of nearest neighbor heuristic.
template<typename TransitionOp>
struct nearest_neighbor final {
  /// Finds next transition
  ANY_EXEC_UNIT vrp::models::Transition operator()(const Context& context,
                                                   int base,
                                                   int from,
                                                   int to,
                                                   int vehicle);

  /// Finds the "nearest" transition for given task and vehicle
  ANY_EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_NEARESTNEIGHBOR_HPP
