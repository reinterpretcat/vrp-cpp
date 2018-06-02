#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/convolutions/SlicedConvolutions.hpp"
#include "algorithms/genetic/Crossovers.hpp"

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::utils;

vrp::models::Convolutions adjusted_cost_difference::operator()(vrp::models::Solution& solution,
                                                               const Settings& settings,
                                                               const Generation& generation) const {
  auto convolutionsLeft =
    create_best_convolutions{}.operator()(solution, settings.convolution, generation.parents.first);

  auto convolutionsRight = create_best_convolutions{}.operator()(solution, settings.convolution,
                                                                 generation.parents.second);

  auto pairs = create_joint_convolutions{}.operator()(solution, settings.convolution,
                                                      convolutionsLeft, convolutionsRight);

  return create_sliced_convolutions{}.operator()(solution, settings.convolution, pairs);
}
