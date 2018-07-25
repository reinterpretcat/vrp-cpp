#include "iterators/Aggregates.hpp"
#include "runtime/Config.hpp"

#include <thrust/detail/use_default.h>
#include <thrust/iterator/detail/discard_iterator_base.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace vrp {
namespace iterators {

template<typename OutputIterator, typename UnaryFunction>
class aggregate_output_iterator;
}

namespace detail {

template<typename UnaryFunction>
class aggregate_output_iterator_proxy {
  UnaryFunction& fun;

public:
  ANY_EXEC_UNIT aggregate_output_iterator_proxy(UnaryFunction fun) : fun(fun) {}

  template<typename T>
  ANY_EXEC_UNIT aggregate_output_iterator_proxy operator=(const T& x) const {
    fun(x);
    return *this;
  }
};

template<typename UnaryFunction, typename OutputIterator>
struct aggregate_output_iterator_base {
  typedef thrust::iterator_adaptor<
    vrp::iterators::aggregate_output_iterator<UnaryFunction, OutputIterator>,
    OutputIterator,
    thrust::use_default,
    thrust::use_default,
    thrust::use_default,
    aggregate_output_iterator_proxy<UnaryFunction>>
    type;
};

}  // namespace detail
}  // namespace vrp
