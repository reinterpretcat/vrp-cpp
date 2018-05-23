#ifndef VRP_ITERATORS_AGGREGATES_HPP
#define VRP_ITERATORS_AGGREGATES_HPP

#include "iterators/detail/Aggregates.inl"

#include <thrust/iterator/discard_iterator.h>

namespace vrp {
namespace iterators {

template<typename UnaryFunction>
class aggregate_output_iterator
  : public vrp::detail::aggregate_output_iterator_base<UnaryFunction>::type
{
  UnaryFunction fun;

public:
  friend class thrust::iterator_core_access;

  typedef typename vrp::detail::aggregate_output_iterator_base<UnaryFunction>::type super_t;

  __host__ __device__ explicit aggregate_output_iterator(UnaryFunction fun) :
    super_t(thrust::make_discard_iterator()), fun(fun) {}

private:
  __host__ __device__ typename super_t::reference dereference() const {
    return vrp::detail::aggregate_output_iterator_proxy<const UnaryFunction>(fun);
  }
};

template<typename UnaryFunction>
aggregate_output_iterator<UnaryFunction> __host__ __device__
make_aggregate_output_iterator(UnaryFunction fun) {
  return aggregate_output_iterator<UnaryFunction>(fun);
}


}  // namespace iterators
}  // namespace vrp

#endif  // VRP_ITERATORS_AGGREGATES_HPP
