#pragma once

#include "models/extensions/problem/Helpers.hpp"
#include "models/problem/Job.hpp"

namespace vrp::models::problem {

/// Returns id of the job.
struct get_job_id final {
  std::string operator()(const Job& job) const {
    return analyze_job<std::string>(job,
                                    [](const std::shared_ptr<const Service>& service) { return service->id; },
                                    [](const std::shared_ptr<const Shipment>& shipment) { return shipment->id; });
  }
};
}