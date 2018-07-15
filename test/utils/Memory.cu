//#include "test_utils/MemoryUtils.hpp"
//#include "test_utils/SettingsFactory.hpp"
//#include "utils/memory/DevicePool.hpp"
#include "runtime/UniquePointer.hpp"

#include <catch/catch.hpp>

using namespace vrp::runtime;

namespace {
__global__ void useDeviceUniquePointer() {
  auto singleInt = make_unique_ptr_value<int>();
  auto arrayInt = make_unique_ptr_data<int>(5);
}

/*__global__ void useDevicePool(thrust::device_ptr<DevicePool> pool) {
  auto ints = pool.get()->ints(3);
  thrust::device_ptr<int> intsPtr = *ints;
  intsPtr[0] = 0;
  intsPtr[1] = 1;
  intsPtr[2] = 2;
}*/

}  // namespace

SCENARIO("Can use device unique pointer on device", "[utils][memory][device][pointer]") {
  useDeviceUniquePointer<<<1, 1, 0>>>();
  cudaStreamSynchronize(0);
}

/*
SCENARIO("Can use device pool on device", "[utils][memory][device][pool]") {
 useDevicePool<<<1, 1, 0>>>(vrp::test::getPool());
 cudaStreamSynchronize(0);
}
*/