#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/genetic/Crossovers.hpp"
#include "algorithms/genetic/Models.hpp"
#include "algorithms/genetic/Mutations.hpp"
#include "algorithms/genetic/Selection.hpp"
#include "streams/input/SolomonReader.hpp"
#include "test_utils/ProblemStreams.hpp"

#include <catch/catch.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <vector>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::genetic;
using namespace vrp::streams;
using namespace vrp::runtime;
using namespace vrp::test;

namespace {

struct init_costs final {
  EXEC_UNIT thrust::pair<int, float> operator()(int index) { return {index, 0}; }
};

}  // namespace

SCENARIO("Can use selection", "[genetic][selection]") {
  auto stream = create_c101_problem_stream{}();
  auto problem = SolomonReader().read(stream, cartesian_distance());
  auto selection = Selection{2, {2, {}}, {2, {}}};
  auto costs = vector<thrust::pair<int, float>>(12);
  thrust::transform(exec_unit, thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(12), costs.begin(), init_costs{});
  auto ctx = EvolutionContext{0, {}, {}, costs};

  select_individuums<empty_crossover, empty_mutator>()(ctx, selection);
}
