#ifndef VRP_ITERATORS_AGGREGATES_HPP
#define VRP_ITERATORS_AGGREGATES_HPP

#include "iterators/detail/Aggregates.inl"
#include "runtime/Config.hpp"

#include <thrust/iterator/discard_iterator.h>

namespace vrp {
namespace iterators {

template<typename UnaryFunction, typename OutputIterator>
class aggregate_output_iterator
  : public vrp::detail::aggregate_output_iterator_base<UnaryFunction, OutputIterator>::type {
  UnaryFunction fun;

public:
  friend class thrust::iterator_core_access;

  typedef typename vrp::detail::aggregate_output_iterator_base<UnaryFunction, OutputIterator>::type
    super_t;

  ANY_EXEC_UNIT explicit aggregate_output_iterator(OutputIterator const& out, UnaryFunction fun) :
    super_t(out), fun(fun) {}

private:
  ANY_EXEC_UNIT typename super_t::reference dereference() const {
    return vrp::detail::aggregate_output_iterator_proxy<UnaryFunction>(fun);
  }
};

template<typename UnaryFunction, typename OutputIterator>
aggregate_output_iterator<UnaryFunction, OutputIterator> ANY_EXEC_UNIT
make_aggregate_output_iterator(OutputIterator out, UnaryFunction fun) {
  return aggregate_output_iterator<UnaryFunction, OutputIterator>(out, fun);
}


}  // namespace iterators
}  // namespace vrp

#endif  // VRP_ITERATORS_AGGREGATES_HPP
