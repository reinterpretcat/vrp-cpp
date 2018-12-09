#pragma once

#include "models/Problem.hpp"
#include "models/common/Timestamp.hpp"
#include "models/problem/Job.hpp"
#include "models/solution/Actor.hpp"

#include <map>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

namespace vrp::algorithms::ruin {

/// Calculates job neighborhood in terms of the cost.
struct JobNeighbourhood final {
  explicit JobNeighbourhood(const models::Problem& problem) {}

  ranges::any_view<models::problem::Job> neighbors(const models::solution::Actor& actor,
                                                   const models::problem::Job& job,
                                                   const models::common::Timestamp time) const {
    // TODO
  }

private:
  std::map<std::string, std::vector<models::problem::Job>> index_;
};
}