#include "runtime/UniquePointer.hpp"

#include <catch/catch.hpp>

using namespace vrp::runtime;

#ifdef RUN_ON_DEVICE

namespace {
__global__ void useUniquePointer() {
  auto singleInt = make_unique_ptr_value<int>();
  auto arrayInt = make_unique_ptr_data<int>(5);
}

}  // namespace

SCENARIO("Can use unique pointer on device", "[utils][memory][device][pointer]") {
  useUniquePointer<<<1, 1, 0>>>();
  cudaStreamSynchronize(0);
}

#endif
