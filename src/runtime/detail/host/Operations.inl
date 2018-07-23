#include <cmath>

namespace vrp {
namespace runtime {

/// Convert a float to a signed integer in round-to-nearest-even mode.
__host__ inline int round(float value) { return std::lround(value); }

}  // namespace runtime
}  // namespace vrp
