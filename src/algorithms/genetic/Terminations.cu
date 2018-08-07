#include "algorithms/genetic/Terminations.hpp"

namespace vrp {
namespace algorithms {
namespace genetic {

bool max_generations::operator()(const EvolutionContext& context) {
  return context.generation > max;
}

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
