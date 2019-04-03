#pragma once

#include "models/Problem.hpp"
#include "models/Solution.hpp"

#include <nlohmann/json.hpp>
#include <sstream>

namespace vrp::test::here {

inline nlohmann::json
getSolutionAsJson(const std::shared_ptr<const models::Problem>& problem,
                  const models::EstimatedSolution& estimatedSolution) {
  std::stringstream ss;
  streams::out::dump_solution_as_here_json{problem}(ss, estimatedSolution);

  return nlohmann::json::parse(ss.str());
}

inline std::vector<nlohmann::json>
getJobActivitiesFromTour(const nlohmann::json& job, const nlohmann::json& tour) {
  using namespace ranges;

  return view::for_each(tour["stops"],
                        [&](const auto& stop) {
                          return view::for_each(stop["activities"], [&](const auto& activity) {
                            return yield_if(activity["jobId"] == job["id"], activity);
                          });
                        }) |
    to_vector;
}
}