#pragma once

#include "models/solution/Activity.hpp"

#include <catch/catch.hpp>
#include <sstream>

namespace vrp::test {

class ActivityMatcher : public Catch::MatcherBase<vrp::models::solution::Activity> {
  vrp::models::solution::Activity activity_;

public:
  explicit ActivityMatcher(vrp::models::solution::Activity activity) : activity_(std::move(activity)) {}

  bool match(const vrp::models::solution::Activity& activity) const override {
    return activity.job == activity_.job && activity.detail.location == activity_.detail.location &&
      activity.schedule.arrival == activity_.schedule.arrival &&
      activity.schedule.departure == activity_.schedule.departure;
  }

  std::string describe() const override {
    // TODO dump activities
    std::ostringstream ss;
    ss << "activities don't match!";
    return ss.str();
  }
};

}  // namespace vrp::test
