#include <chrono>

namespace vrp {
namespace runtime {

/// Returns randomization seed
EXEC_UNIT inline unsigned int random_seed() {
  return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
}

}  // namespace runtime
}  // namespace vrp