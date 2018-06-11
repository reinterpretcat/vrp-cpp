#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "algorithms/genetic/Crossovers.hpp"

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::utils;

__device__ void adjusted_cost_difference::operator()(const Settings& settings,
                                                     const Generation& generation) const {
  auto left = create_best_convolutions{solution, pool}.operator()(settings.convolution,
                                                                  generation.parents.first);
  auto right = create_best_convolutions{solution, pool}.operator()(settings.convolution,
                                                                   generation.parents.second);
  auto pairs =
    create_joint_convolutions{solution, pool}.operator()(settings.convolution, left, right);

  auto convolutions =
    create_sliced_convolutions{solution, pool}.operator()(settings.convolution, pairs);

  // TODO
}
