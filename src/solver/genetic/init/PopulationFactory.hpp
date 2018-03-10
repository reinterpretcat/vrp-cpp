#ifndef VRP_SOLVER_GENETIC_POPULATIONFACTORY_HPP
#define VRP_SOLVER_GENETIC_POPULATIONFACTORY_HPP

#include "models/Problem.hpp"
#include "models/Resources.hpp"
#include "models/Tasks.hpp"
#include "solver/genetic/Settings.hpp"

namespace vrp {
namespace genetic {

/// Creates initial population based on problem, resources and settings.
vrp::models::Tasks createPopulation(const vrp::models::Problem &problem,
                                    const vrp::models::Resources &resources,
                                    const vrp::genetic::Settings &settings);
}
}

#endif //VRP_SOLVER_GENETIC_POPULATIONFACTORY_HPP
