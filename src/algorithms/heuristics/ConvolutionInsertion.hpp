#ifndef VRP_HEURISTICS_CONVOLUTIONINSERTION_HPP
#define VRP_HEURISTICS_CONVOLUTIONINSERTION_HPP

#include "algorithms/heuristics/Models.hpp"
#include "runtime/UniquePointer.hpp"

namespace vrp {
namespace algorithms {
namespace heuristics {

/// A helper class which provides the way to get first convolutions and them other customers.
struct find_convolution_customer {
  EXEC_UNIT find_convolution_customer(const Context& context, int base);
  EXEC_UNIT vrp::runtime::variant<int, vrp::models::Convolution> operator()();

private:
  const Context& context;
  int base;
  vrp::runtime::unique_ptr<vrp::runtime::vector_ptr<int>> ids;
  int last;
};

/// Implements algorithm of convolution insertion heuristic.
template<typename TransitionOp>
struct convolution_insertion final {
  /// Populates individuum with given index starting from task defined by shift.
  EXEC_UNIT void operator()(const Context& context, int index, int shift);
};

}  // namespace heuristics
}  // namespace algorithms
}  // namespace vrp

#endif  // VRP_HEURISTICS_CONVOLUTIONINSERTION_HPP
