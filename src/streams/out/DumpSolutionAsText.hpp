#pragma once

#include "models/Solution.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Helpers.hpp"

#include <ostream>
#include <range/v3/all.hpp>

namespace vrp::streams::out {

/// Dumps solution to ostream in simplest human readable way.
struct dump_solution_as_text final {
  void operator()(std::ostream& out, const models::EstimatedSolution& es) const {
    out << "\t\tactual cost:" << es.second.actual << " + penalties: " << es.second.penalty
        << "\n\t\ttotal routes:" << es.first->routes.size();
    ranges::for_each(es.first->routes, [&](const auto& route) {
      out << "\n\t\tvehicle " << models::problem::get_vehicle_id{}(*route->actor->vehicle) << ", customers:";
      ranges::for_each(route->tour.activities(), [&](const auto& a) {
        auto job = models::solution::retrieve_job{}(*a);
        if (job.has_value()) { out << " " << models::problem::get_job_id{}(job.value()); }
      });
    });
    out << std::endl;
  }
};
}