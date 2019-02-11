#pragma once

#include <range/v3/all.hpp>

namespace vrp::utils {

/// Accumulates values while predicate returns true.
template<typename View, typename T, typename Pred, typename Accumulator>
inline T&&
accumulate_while(const View& view, T&& value, const Pred& predicate, const Accumulator& accumulator) {
  for (auto it = ranges::begin(view); it != ranges::end(view); ++it) {
    if (!predicate(value)) break;
    value = std::move(accumulator(value, *it));
  }
  return std::move(value);
}
}