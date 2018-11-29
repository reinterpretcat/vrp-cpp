#pragma once

#include "models/problem/Fleet.hpp"
#include "models/problem/Job.hpp"

#include <memory>
#include <vector>

namespace vrp::models {

/// Defines VRP problem.
struct Problem :final {

  std::shared_ptr<Fleet> fleet;

  std::vector<problem::Job> jobs;
};

}
