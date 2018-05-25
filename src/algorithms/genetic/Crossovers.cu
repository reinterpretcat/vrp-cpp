#include "algorithms/convolutions/BestConvolutions.hpp"
#include "algorithms/convolutions/JointConvolutions.hpp"
#include "algorithms/genetic/Crossovers.hpp"

using namespace vrp::algorithms::convolutions;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::utils;

void adjusted_cost_difference::operator()(const Problem& problem,
                                          Tasks& tasks,
                                          const Settings& settings,
                                          const Generation& generation) const {
  auto convolutionsLeft = create_best_convolutions{}.operator()(
    problem, tasks, settings.convolution, generation.parents.first);

  auto convolutionsRight = create_best_convolutions{}.operator()(
    problem, tasks, settings.convolution, generation.parents.second);

  auto pairs = create_joint_convolutions{}.operator()(problem, tasks, settings.convolution,
                                                      convolutionsLeft, convolutionsRight);
}
