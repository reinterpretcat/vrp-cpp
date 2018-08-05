#ifndef VRP_ITERATORS_SEQUENTIAL_HPP
#define VRP_ITERATORS_SEQUENTIAL_HPP

#include "runtime/Config.hpp"

namespace vrp {
namespace iterators {

/// Iterates sequentially
template<typename Iterator, typename Action>
ANY_EXEC_UNIT void for_seq(Iterator first, Iterator last, Action action) {
  for (int i = 0; first != last; ++first, ++i) {
    action(i, *first);
  }
}

}  // namespace iterators
}  // namespace vrp

#endif  // VRP_ITERATORS_SEQUENTIAL_HPP
