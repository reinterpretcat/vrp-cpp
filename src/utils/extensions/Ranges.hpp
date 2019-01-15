#pragma once

#include <algorithm>
#include <range/v3/all.hpp>

namespace vrp::utils {

/// Accumulates values while predicate returns true.
template<typename View, typename T, typename Pred, typename Accumulator>
T
accumulate_while(View view, T value, Pred predicate, Accumulator accumulator) {
  return ranges::accumulate(
    view | ranges::view::take_while([&predicate, &value](const auto&) { return predicate(value); }),
    value,
    [&accumulator, &value](const auto& acc, const auto& item) {
      value = accumulator(acc, item);
      return value;
    });
}
}