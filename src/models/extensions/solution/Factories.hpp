#pragma once

#include "models/solution/Activity.hpp"

#include <memory>

namespace vrp::models::solution {

class build_activity {
public:
  build_activity& withSchedule(common::Schedule&& schedule) {
    activity_.schedule = schedule;
    return *this;
  }

  build_activity& withLocation(common::Location&& location) {
    activity_.location = location;
    return *this;
  }

  build_activity& withJob(std::shared_ptr<const vrp::models::problem::Job> job) {
    activity_.job = job;
    return *this;
  }

  Activity&& owned() { return std::move(activity_); }

private:
  Activity activity_;
};

}  // namespace vrp::models::solution
