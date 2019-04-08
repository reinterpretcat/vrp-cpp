#if _MSC_VER
#define EXPORT_API __declspec(dllexport)
#elif _GCC
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT_API
#endif

#include "Solver.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "streams/out/json/HereSolutionJson.hpp"

#include <sstream>
#include <string>

/// Called when solution is created.
typedef void OnSuccess(const char* solution);

/// Called when solution cannot be created.
typedef void OnError(const char* error);

extern "C" {
void EXPORT_API
solve(const char* jsonProblem, const char* format, OnSuccess* onSuccess, OnError* onError) {
  // TODO extract this functionality to header and reuse it in SolverExe
  auto solver = vrp::Solver<vrp::algorithms::refinement::create_refinement_context<>,
                            vrp::algorithms::refinement::select_best_solution,
                            vrp::algorithms::refinement::ruin_and_recreate_solution<>,
                            vrp::algorithms::refinement::GreedyAcceptance<>,
                            vrp::algorithms::refinement::MaxIterationCriteria,
                            vrp::algorithms::refinement::log_to_nothing>{};

  if (std::strcmp(format, "here") == 0) {
    std::stringstream ss;
    ss << jsonProblem;

    auto problem = vrp::streams::in::read_here_json_type{}(ss);
    auto solution = solver(problem);

    ss.str("");
    ss.clear();

    vrp::streams::out::dump_solution_as_here_json{problem}(ss, solution);
    auto jsonSolution = ss.str();
    onSuccess(jsonSolution.data());
  } else {
    // TODO add all supported inputs
    onError("Not supported");
  }
}
}
