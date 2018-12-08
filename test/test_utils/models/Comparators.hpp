#pragma once

#include "models/common/Schedule.hpp"

namespace vrp::test {

struct compare_schedules final {
  bool operator()(const models::common::Schedule& lhs, const models::common::Schedule& rhs) const {
    return lhs.arrival == rhs.arrival && lhs.departure == rhs.departure;
  }
};
}