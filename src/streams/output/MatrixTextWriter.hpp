#ifndef VRP_STREAMS_MATRIXTEXTWRITER_HPP
#define VRP_STREAMS_MATRIXTEXTWRITER_HPP

#include "models/Solution.hpp"

namespace vrp {
namespace streams {

/// Writes solution into output stream as a text in matrix format.
class MatrixTextWriter final {
public:
  /// Writes solution as text into stream.
  static void write(std::ostream& out, const vrp::models::Solution& solution);

  /// Writes tasks as text into stream.
  static void write(std::ostream& out, const vrp::models::Tasks& tasks);
};

}  // namespace streams
}  // namespace vrp

#endif  // VRP_STREAMS_MATRIXTEXTWRITER_HPP
