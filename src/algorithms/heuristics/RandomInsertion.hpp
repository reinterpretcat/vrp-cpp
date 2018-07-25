#ifndef VRP_HEURISTICS_RANDOMINSERTION_HPP
#define VRP_HEURISTICS_RANDOMINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of random insertion heuristic.
template<typename TransitionOp>
struct random_insertion final {
  /// Populates individuum with given index starting from task defined by shift.
  ANY_EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_RANDOMINSERTION_HPP
