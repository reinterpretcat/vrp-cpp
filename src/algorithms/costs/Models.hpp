#ifndef VRP_ALGORITHMS_COSTS_MODELS_HPP
#define VRP_ALGORITHMS_COSTS_MODELS_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace algorithms {
namespace costs {

/// Contains model shadows.
struct Model final {
  float total;
  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;
};

}
}
}

#endif //VRP_ALGORITHMS_COSTS_MODELS_HPP
