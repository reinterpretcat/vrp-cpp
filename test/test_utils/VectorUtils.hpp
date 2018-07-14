#ifndef VRP_UTILS_VECTORUTILS_HPP
#define VRP_UTILS_VECTORUTILS_HPP

#include "runtime/Config.hpp"
#include <vector>

namespace vrp {
namespace test {

/// Creates vrp vector from initializer list.
template<typename T>
inline vrp::runtime::vector<T> create(const std::initializer_list<T>& list) {
  thrust::host_vector<T> buffer(list.begin(), list.end());
  return vrp::runtime::vector<T>(buffer.begin(), buffer.end());
}

/// Copies vrp vector to std vector.
template<typename T>
std::vector<T> copy(const vrp::runtime::vector<T>& v) {
  return std::vector<T>(v.begin(), v.end());
};

/// Copies n-first items of vrp vector to std vector.
template<typename T>
std::vector<T> copy(const vrp::runtime::vector<T>& v, std::size_t n) {
  return std::vector<T>(v.begin(), v.begin() + n);
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_UTILS_VECTORUTILS_HPP
