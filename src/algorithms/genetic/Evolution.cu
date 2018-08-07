#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Listeners.hpp"
#include "algorithms/genetic/Terminations.hpp"

using namespace vrp::algorithms::genetic;
using namespace vrp::models;

namespace vrp {
namespace algorithms {
namespace genetic {

template<typename TerminationCriteria, typename GenerationListener>
void run_evolution<TerminationCriteria, GenerationListener>::operator()(const Problem& problem,
                                                                        const Settings& settings) {}

// NOTE explicit specialization to make linker happy.
template class run_evolution<max_generations, empty_listener>;

}  // namespace genetic
}  // namespace algorithms
}  // namespace vrp
