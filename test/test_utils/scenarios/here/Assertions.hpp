#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "streams/out/json/HereSolutionJson.hpp"

#include <catch/catch.hpp>
#include <nlohmann/json.hpp>
#include <sstream>

namespace vrp::test::here {

inline void
assertSolution(const std::shared_ptr<const models::Problem>& problem,
               const models::EstimatedSolution& estimatedSolution,
               const std::string& expected) {
  std::stringstream ss;
  streams::out::dump_solution_as_here_json{problem}(ss, estimatedSolution);

  REQUIRE(nlohmann::json::parse(ss.str()) == nlohmann::json::parse(expected));
}
}