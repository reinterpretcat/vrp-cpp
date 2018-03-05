#ifndef VRP_UTILS_PRINT_HPP
#define VRP_UTILS_PRINT_HPP

#include <thrust/host_vector.h>
#include <iostream>

namespace vrp {
namespace utils {

/// Prints host vector to standard output.
template<typename T>
void print(const thrust::host_vector<T> &v) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, ", "));
}

}
}

#endif //VRP_UTILS_PRINT_HPP
