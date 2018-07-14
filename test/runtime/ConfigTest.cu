#include "runtime/Config.hpp"

#include <catch/catch.hpp>

using namespace vrp::runtime;

namespace {
struct generator {
  EXEC_UNIT int operator()() const { return 1; }
};

struct checker {
  EXEC_UNIT void operator()(int value) { assert(value == 1); }
};

}  // namespace

TEST_CASE("Can use runtime config.", "[runtime][config]") {
  size_t size = 1 << 10;
  checker check{};
  vrp::runtime::vector<int> input(size);

  auto pointer = input.data();

  thrust::generate(input.begin(), input.end(), generator{});
  thrust::sort(exec_unit, input.begin(), input.end());
  thrust::for_each(exec_unit, pointer, pointer  + size, check);
}
