#ifndef VRP_HEURISTICS_DUMMY_HPP
#define VRP_HEURISTICS_DUMMY_HPP

#include "algorithms/heuristics/Models.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// A dummy implementation of heuristic which returns an invalid transition and cost.
template<typename TransitionOp>
struct dummy final {
  ANY_EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_DUMMY_HPP
