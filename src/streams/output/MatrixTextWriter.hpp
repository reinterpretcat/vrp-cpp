#ifndef VRP_STREAMS_MATRIXTEXTWRITER_HPP
#define VRP_STREAMS_MATRIXTEXTWRITER_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace streams {

/// Writes solution into output stream as a text in matrix format.
class MatrixTextWriter final {
public:
  /// Writes text to stream.
  void write(std::ostream& out,
             const vrp::models::Problem& problem,
             const vrp::models::Tasks& tasks);
};

}  // namespace streams
}  // namespace vrp

#endif  // VRP_STREAMS_MATRIXTEXTWRITER_HPP
