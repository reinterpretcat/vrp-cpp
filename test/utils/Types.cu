#include "utils/types/DeviceVariant.hpp"

#include <catch/catch.hpp>
#include <thrust/execution_policy.h>

using namespace vrp::utils;

namespace {

__global__ void useDeviceVariant() {
  device_variant<bool, int> value;

  value.set<bool>(true);
  assert(value.is<bool>());
  assert(!value.is<int>());
  assert(value.get<bool>() == true);

  value.set<int>(7);
  assert(!value.is<bool>());
  assert(value.is<int>());
  assert(value.get<int>() == 7);
}

}  // namespace

SCENARIO("Can use device variant on device", "[utils][types][device][variant]") {
  useDeviceVariant<<<1, 1, 0>>>();
  cudaStreamSynchronize(0);
}
