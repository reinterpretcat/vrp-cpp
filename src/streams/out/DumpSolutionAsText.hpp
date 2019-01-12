#pragma once

#include "models/Solution.hpp"

#include <ostream>
#include <range/v3/all.hpp>

namespace vrp::streams::out {

/// Dumps solution to ostream in simplest human readable way.
struct dump_solution_as_text final {
  void operator()(std::ostream& out, const models::EstimatedSolution& es) const {
    out << "\t\tactual cost:" << es.second.actual << " + penalties: " << es.second.penalty
        << "\n\t\ttotal routes:" << es.first->routes.size();
    ranges::for_each(es.first->routes, [&](const auto& route) {
      out << "\n\t\tvehicle " << route->actor->vehicle->id << ", customers:";
      ranges::for_each(route->tour.activities(), [&](const auto& a) {
        // TODO print activities without job
        assert(a->job.has_value());
        out << " " << getId(a->job.value());
      });
    });
    out << std::endl;
  }

private:
  std::string getId(const models::problem::Job& job) const {
    return utils::mono_result(const_cast<models::problem::Job&>(job).visit(
      ranges::overload([](const std::shared_ptr<const models::problem::Service>& service) { return service->id; },
                       [](const std::shared_ptr<const models::problem::Shipment>& shipment) { return shipment->id; })));
  }
};
}