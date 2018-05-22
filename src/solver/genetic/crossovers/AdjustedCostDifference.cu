#include "solver/genetic/crossovers/AdjustedCostDifference.hpp"

using namespace vrp::genetic;
using namespace vrp::models;
using namespace vrp::utils;

using Generation = adjusted_cost_difference::Generation;

void adjusted_cost_difference::operator()(const Problem& problem,
                                          Tasks& tasks,
                                          const Settings& settings,
                                          const Generation& generation,
                                          Pool& pool) const {
  // TODO
}
