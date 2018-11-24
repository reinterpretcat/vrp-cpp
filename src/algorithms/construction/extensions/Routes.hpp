#pragma once

#include "models/common/Timestamp.hpp"
#include "models/extensions/problem/Factories.hpp"
#include "models/extensions/solution/Factories.hpp"
#include "models/solution/Actor.hpp"

namespace vrp::algorithms::construction {

/// Creates start/end activities for given actor and its departure.
inline std::pair<models::solution::Tour::Activity, models::solution::Tour::Activity>
waypoints(const models::solution::Actor& actor, models::common::Timestamp departure) {
  using namespace vrp::utils;
  using namespace vrp::models;

  const auto& detail = actor.detail;

  // create start/end for new vehicle
  auto start = solution::build_activity{}
                 .type(solution::Activity::Type::Start)
                 .detail({detail.start, 0, {detail.time.start, std::numeric_limits<common::Timestamp>::max()}})
                 .schedule({detail.time.start, departure})  //
                 .shared();
  auto end = solution::build_activity{}
               .type(solution::Activity::Type::End)
               .detail({detail.end.value_or(detail.time.start), 0, {0, detail.time.end}})
               .schedule({0, detail.time.end})  //
               .shared();

  return {start, end};
}
}