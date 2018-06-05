#ifndef VRP_TESTUTILS_MEMORYUTILS_HPP
#define VRP_TESTUTILS_MEMORYUTILS_HPP

#include "utils/memory/DevicePool.hpp"

namespace vrp {
namespace test {

/// Returns device pool.
inline vrp::utils::DevicePool::Pointer getPool() {
  static auto pool = vrp::utils::DevicePool::create(1, 4, 100);
  return *pool;
}

}  // namespace test
}  // namespace vrp

#endif  // VRP_TESTUTILS_MEMORYUTILS_HPP