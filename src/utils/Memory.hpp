#ifndef VRP_UTILS_MEMORY_HPP
#define VRP_UTILS_MEMORY_HPP

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

namespace vrp {
namespace utils {

/// Allocates a new buffer in memory initializing with given value.
template<typename T>
inline thrust::device_ptr<T> allocate(const T& value) {
  thrust::device_ptr<T> pValue = thrust::device_malloc<T>(1);
  *pValue = value;
  return pValue;
}

/// Releases buffer returning its value.
template<typename T>
inline T release(thrust::device_ptr<T>& buffer) {
  T value = *buffer;
  thrust::device_free(buffer);
  return value;
}

}  // namespace utils
}  // namespace vrp

#endif  // VRP_UTILS_MEMORY_HPP