#pragma once

#include "models/problem/Job.hpp"
#include "utils/extensions/Variant.hpp"

#include <range/v3/all.hpp>

namespace vrp::test {

/// Returns job id.
struct get_job_id final {
  std::string operator()(const models::problem::Job& job) {
    return utils::mono_result(const_cast<models::problem::Job&>(job).visit(
      ranges::overload([](const std::shared_ptr<const models::problem::Service>& service) { return service->id; },
                       [](const std::shared_ptr<const models::problem::Shipment>& shipment) { return shipment->id; })));
  }
};
}