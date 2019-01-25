#pragma once

#include "models/extensions/problem/Helpers.hpp"

#include <cstdint>
#include <memory>

namespace vrp::models::problem {

struct is_the_same_jobs;

/// Compares jobs using less.
struct compare_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const {
    return getPointerAsIntPtr(lhs) < getPointerAsIntPtr(rhs);
  }

private:
  friend is_the_same_jobs;
  static std::uintptr_t getPointerAsIntPtr(const problem::Job& job) {
    return analyze_job<std::uintptr_t>(
      job,
      [](const std::shared_ptr<const Service>& service) { return reinterpret_cast<std::uintptr_t>(service.get()); },
      [](const std::shared_ptr<const Shipment>& shipment) { return reinterpret_cast<std::uintptr_t>(shipment.get()); });
  }
};

/// Checks whether two jobs are the same.
struct is_the_same_jobs final {
  bool operator()(const problem::Job& lhs, const problem::Job& rhs) const {
    return compare_jobs::getPointerAsIntPtr(lhs) == compare_jobs::getPointerAsIntPtr(rhs);
  }
};

}  // namespace vrp::models::problem
