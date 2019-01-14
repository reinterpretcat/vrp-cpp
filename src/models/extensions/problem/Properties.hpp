#pragma once

#include "models/problem/Job.hpp"
#include "utils/extensions/Variant.hpp"

namespace vrp::models::problem {

/// Returns id of the job.
struct get_job_id final {
  std::string operator()(const Job& job) const {
    return utils::mono_result(const_cast<Job&>(job).visit(
      ranges::overload([](const std::shared_ptr<const Service>& service) { return service->id; },
                       [](const std::shared_ptr<const Shipment>& shipment) { return shipment->id; })));
  }
};
}