#pragma once

#include <range/v3/all.hpp>

namespace vrp::utils {

/// Accumulates values while predicate returns true.
template<typename View, typename T, typename Pred, typename Accumulator>
inline T&&
accumulate_while(const View& view, T&& value, const Pred& predicate, const Accumulator& accumulator) {
  ranges::find_if(const_cast<View&>(view), [&value, &predicate, &accumulator](const auto& item) {
    value = std::move(accumulator(value, item));
    return !predicate(value);
  });
  return std::move(value);
}
}