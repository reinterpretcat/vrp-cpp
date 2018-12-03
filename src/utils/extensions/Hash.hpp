#pragma once

#include <cstddef>

namespace vrp::utils {

template<typename T>
struct hash_combine {
  T value;
  std::size_t operator()(std::size_t seed) const { return value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }
};

template<typename T>
std::size_t
operator|(std::size_t seed, const hash_combine<T>& hasher) {
  return hasher(seed);
}
}