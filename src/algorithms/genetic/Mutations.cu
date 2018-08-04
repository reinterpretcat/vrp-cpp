#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/heuristics/Models.hpp"

using namespace vrp::algorithms::heuristics;

namespace {}

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TransitionOp>
ANY_EXEC_UNIT void create_mutant<TransitionOp>::operator()(const Mutation& mutation) const {
  assert(false);
}

/// NOTE make linker happy.
template class create_mutant<TransitionOperator>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
