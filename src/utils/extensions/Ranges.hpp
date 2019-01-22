#pragma once

#include <range/v3/all.hpp>

namespace vrp::utils {

/// Accumulates values while predicate returns true.
template<typename View, typename T, typename Pred, typename Accumulator>
T
accumulate_while(const View& view, T value, Pred predicate, Accumulator accumulator) {
  for (auto it = ranges::begin(view); it != ranges::end(view); ++it) {
    if (!predicate(value)) break;
    value = accumulator(value, *it);
  }
  return value;
}
}