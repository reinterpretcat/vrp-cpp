#include "Solver.hpp"
#include "streams/in/scientific/LiLim.hpp"
#include "streams/in/scientific/Solomon.hpp"

#include <algorithms/refinement/logging/LogToConsole.hpp>
#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

int
main(int argc, char* argv[]) {
  if (argc < 2) throw std::invalid_argument("Missing path to solomon problem.");

  auto stream = std::fstream(argv[1], std::ios::in);
  auto problem = argc > 2 && std::string(argv[2]) == "lilim"
    ? read_li_lim_type<cartesian_distance>{}.operator()(stream)
    : read_solomon_type<cartesian_distance>{}.operator()(stream);

  auto solver = vrp::Solver<vrp::algorithms::refinement::create_refinement_context<>,
                            vrp::algorithms::refinement::select_best_solution,
                            vrp::algorithms::refinement::ruin_and_recreate_solution<>,
                            vrp::algorithms::refinement::GreedyAcceptance<>,
                            vrp::algorithms::refinement::MaxIterationCriteria,
                            vrp::algorithms::refinement::log_to_console>{};

  auto estimatedSolution = solver(problem);
}