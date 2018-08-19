#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Strategies.hpp"

#include <catch/catch.hpp>
#include <thrust/iterator/counting_iterator.h>

using namespace vrp::algorithms::genetic;
using namespace vrp::runtime;

SCENARIO("Can use linear strategy's selection and next", "[genetic][strategy]") {
  auto strategy = LinearStrategy{ {16} };
  auto ctx = EvolutionContext{0,
                              {},
                              vector<thrust::pair<int, float>>(static_cast<size_t>(16)),
                              thrust::minstd_rand()};

  std::for_each(thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(5), [&](int i) {
    strategy.selection(ctx);
    strategy.next(ctx);
  });
}
