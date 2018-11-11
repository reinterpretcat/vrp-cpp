#pragma once

#include <range/v3/utility/variant.hpp>

namespace vrp::utils {

/// Extracts result from variant which holds two values of the same type.
/// TODO use variadic templates to allow variant with more than two values?
template<typename T>
inline auto
mono_result(const ranges::variant<T, T>& v) {
  return v.index() == 0 ? ranges::get<0>(v) : ranges::get<1>(v);
}
}
