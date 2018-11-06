#pragma once

#include "models/common/Location.hpp"
#include "models/problem/plan/Job.hpp"

#include <optional>

namespace vrp::models::problem::plan {

struct Break final : public Job {

  std::optional<vrp::models::common::Location> location;

  void accept(JobVisitor &visitor) const override {
    visitor.visit(*this);
  }
};

}
