#ifndef VRP_HEURISTICS_CHEAPESTINSERTION_HPP
#define VRP_HEURISTICS_CHEAPESTINSERTION_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"

namespace vrp {
namespace heuristics {

/// Implements algorithm of cheapest insertion heuristic.
struct CheapestInsertion final {

  vrp::models::Problem::Shadow problem;
  vrp::models::Tasks::Shadow tasks;

  void operator()(int fromTask) {
    auto vehicle = tasks.vehicles[fromTask];
    auto fromCustomer = tasks.ids[fromTask];
  }
};

}
}

#endif //VRP_HEURISTICS_CHEAPESTINSERTION_HPP