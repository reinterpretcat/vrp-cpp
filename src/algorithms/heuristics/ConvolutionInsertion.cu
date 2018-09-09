#include "algorithms/convolutions/Models.hpp"
#include "algorithms/heuristics/CheapestInsertion.hpp"
#include "algorithms/heuristics/ConvolutionInsertion.hpp"

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::heuristics;
using namespace vrp::models;
using namespace vrp::runtime;

namespace {

/// Alias for customer variant.
using Customer = variant<int, Convolution>;

struct sort_customers final {
  const Context& context;
  int base;

  EXEC_UNIT bool operator()(int left, int right) {
    Plan leftPlan = *(context.tasks.plan + base + left);
    Plan rightPlan = *(context.tasks.plan + base + right);

    if (leftPlan.hasConvolution()) return true;
    if (rightPlan.hasConvolution()) return false;

    int leftStart = *(context.problem.customers.starts + base + left);
    int rightStart = *(context.problem.customers.starts + base + right);

    return leftStart < rightStart;
  }
};

}  // namespace

namespace vrp {
namespace algorithms {
namespace heuristics {

EXEC_UNIT find_convolution_customer::find_convolution_customer(const Context& context, int base) :
  context(context), base(base),
  ids(make_unique_ptr_data<int>(static_cast<size_t>(context.problem.size))), last(1) {
  thrust::sequence(exec_unit_policy{}, *ids.get(), *ids.get() + context.problem.size, 0);

  thrust::sort(exec_unit_policy{}, *ids.get() + 1, *ids.get() + context.problem.size,
               sort_customers{context, base});
}

EXEC_UNIT Customer find_convolution_customer::operator()() {
  while (last < context.problem.size) {
    int index = *(*ids.get() + last);
    Plan plan = *(context.tasks.plan + base + index);

    if (!plan.isAssigned())
      return plan.hasConvolution()
               ? Customer::create<Convolution>(*(context.convolutions + plan.convolution()))
               : Customer::create<int>(index);
    last++;
  }

  return Customer::create<int>(-1);
}

template<typename TransitionOp>
void convolution_insertion<TransitionOp>::operator()(const Context& context, int index, int shift) {
  cheapest_insertion<TransitionOp, find_convolution_customer>{}(context, index, shift);
}

/// NOTE make linker happy.
template class convolution_insertion<TransitionOperator>;

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp
