#ifndef VRP_HEURISTICS_CHEAPESTINSERTION_HPP
#define VRP_HEURISTICS_CHEAPESTINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// Implements algorithm of cheapest insertion heuristic with
/// generalized customer selector.
template<typename TransitionOp, typename CustomerSelector>
struct cheapest_insertion final {
  /// Populates individuum with given index starting from task defined by shift.
  EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_CHEAPESTINSERTION_HPP
