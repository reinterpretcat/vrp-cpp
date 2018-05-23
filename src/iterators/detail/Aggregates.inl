#include "iterators/Aggregates.hpp"

#include <thrust/detail/use_default.h>
#include <thrust/iterator/detail/discard_iterator_base.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace vrp {

namespace iterators {
template<typename UnaryFunction>
class aggregate_output_iterator;
}

namespace detail {

template<typename UnaryFunction>
class aggregate_output_iterator_proxy
{
  UnaryFunction& fun;

public:
  __host__ __device__ aggregate_output_iterator_proxy(UnaryFunction& fun) : fun(fun) {}

  template<typename T>
  __host__ __device__ aggregate_output_iterator_proxy operator=(const T& x) const {
    return *this;
  }
};

template<typename UnaryFunction>
struct aggregate_output_iterator_base {
  typedef thrust::iterator_adaptor<vrp::iterators::aggregate_output_iterator<UnaryFunction>,
                                   thrust::detail::discard_iterator_base<thrust::use_default>::type,
                                   thrust::use_default,
                                   thrust::use_default,
                                   thrust::use_default,
                                   aggregate_output_iterator_proxy<const UnaryFunction>>
    type;
};

}  // namespace detail
}  // namespace vrp
