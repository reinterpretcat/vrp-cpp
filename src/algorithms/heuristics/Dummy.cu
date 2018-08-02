#include "algorithms/heuristics/Dummy.hpp"

#include <thrust/tuple.h>

using namespace vrp::algorithms::heuristics;

namespace vrp {
namespace algorithms {
namespace heuristics {

template<typename TransitionOp>
ANY_EXEC_UNIT void dummy<TransitionOp>::operator()(const Context& context, int index, int shift){};

/// NOTE make linker happy.
template class dummy<TransitionOperator>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
