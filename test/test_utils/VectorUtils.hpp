#ifndef VRP_UTILS_VECTORUTILS_HPP
#define VRP_UTILS_VECTORUTILS_HPP

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>

namespace vrp {
namespace test {

/// Copies host vector to std vector.
template<typename T>
std::vector<T> copy(const thrust::host_vector<T> &v) {
  return std::vector<T>(v.begin(), v.end());
}

/// Copies device vector to std vector.
template<typename T>
std::vector<T> copy(const thrust::device_vector<T> &v) {
  thrust::host_vector<T> hv = v;
  return copy(hv);
}

}
}

#endif //VRP_UTILS_VECTORUTILS_HPP
