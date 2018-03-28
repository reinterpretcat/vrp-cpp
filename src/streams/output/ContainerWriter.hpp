#ifndef VRP_STREAMS_CONTAINERWRITER_HPP
#define VRP_STREAMS_CONTAINERWRITER_HPP

#include <thrust/host_vector.h>
#include <ostream>

namespace vrp {
namespace streams {

/// Writes container to output stream with delimiter.
template<typename Container>
void write(std::ostream &out, const Container &v, const char *delimiter) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<typename Container::value_type>(out, delimiter));
}

}
}

#endif //VRP_STREAMS_CONTAINERWRITER_HPP
