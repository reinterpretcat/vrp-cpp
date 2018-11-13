#pragma once

#include "models/problem/Job.hpp"
#include "utils/extensions/Variant.hpp"

#include <memory>

namespace vrp::models::problem {

/// Compares jobs.
struct compare_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const {
    static const auto fun =
      ranges::overload([](const std::shared_ptr<const Service>& service) { return service->id; },
                       [](const std::shared_ptr<const Shipment>& shipment) { return shipment->id; });

    auto left = utils::mono_result(const_cast<problem::Job&>(lhs).visit(fun));
    auto right = utils::mono_result(const_cast<problem::Job&>(rhs).visit(fun));

    return left < right;
  }
};

}  // namespace vrp::models::problem
