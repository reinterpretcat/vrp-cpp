#pragma once

#include "AlgorithmDefinition.hpp"
#include "Solver.hpp"
#include "streams/in/json/HereProblemJson.hpp"
#include "streams/in/json/RichProblemJson.hpp"
#include "streams/in/scientific/LiLim.hpp"
#include "streams/in/scientific/Solomon.hpp"
#include "streams/out/json/HereSolutionJson.hpp"
#include "streams/out/text/DumpSolutionAsText.hpp"

#include <iostream>
#include <sstream>

namespace vrp::example {

/// Solves the problem from input stream storing result in output stream.
struct solve_based_on_type final {
  void operator()(const char* inType, std::istream& in, const char* outType, std::ostream& out) const {
    auto solver = Solver<AlgorithmDefinition>{};

    auto problem = readProblem(inType, in);

    auto solution = solver(problem);

    writeSolution(problem, solution, outType, out);
  }

private:
  std::shared_ptr<models::Problem> readProblem(const char* inType, std::istream& in) const {
    std::stringstream ss;

    std::copy(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), std::ostreambuf_iterator<char>(ss));

    if (std::strcmp(inType, "here") == 0) return vrp::streams::in::read_here_json_type{}(ss);
    if (std::strcmp(inType, "rich") == 0) return vrp::streams::in::read_rich_json_type{}(ss);

    if (std::strcmp(inType, "lilim") == 0) return vrp::streams::in::read_li_lim_type{}(ss);
    if (std::strcmp(inType, "solomon") == 0) return vrp::streams::in::read_solomon_type{}(ss);

    throw std::invalid_argument(std::string("Unknown input stream type: '") + inType + "'.");
  }

  void writeSolution(const std::shared_ptr<models::Problem>& problem,
                     const models::EstimatedSolution& solution,
                     const char* outType,
                     std::ostream& out) const {
    // TODO move problem to function parameters
    if (std::strcmp(outType, "here") == 0) return vrp::streams::out::dump_solution_as_here_json{problem}(out, solution);
    if (std::strcmp(outType, "text") == 0) return vrp::streams::out::dump_solution_as_text{}(out, solution);

    throw std::invalid_argument(std::string("Unknown output stream type: '") + outType + "'.");
  }
};
}