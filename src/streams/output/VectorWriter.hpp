#ifndef VRP_STREAMS_VECTORWRITER_HPP
#define VRP_STREAMS_VECTORWRITER_HPP

#include <thrust/host_vector.h>
#include <ostream>

namespace vrp {
namespace streams {

/// Writes host vector to output stream with delimiter.
template<typename T>
void write(std::ostream &out, const thrust::host_vector<T> &v, const char *delimiter) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, delimiter));
}

}
}

#endif //VRP_STREAMS_VECTORWRITER_HPP
