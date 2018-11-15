#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "test_utils/models/Factories.hpp"

namespace vrp::test {

struct test_build_insertion_route_context : public vrp::algorithms::construction::build_insertion_route_context {
  explicit test_build_insertion_route_context() : vrp::algorithms::construction::build_insertion_route_context() {
    withActor(DefaultActor)
      .withRoute(test_build_route{}.shared())
      .withTime(DefaultTime)
      .withState(std::make_shared<vrp::algorithms::construction::InsertionRouteState>());
  }
};

struct test_build_insertion_activity_context : public vrp::algorithms::construction::build_insertion_activity_context {
  explicit test_build_insertion_activity_context() : vrp::algorithms::construction::build_insertion_activity_context() {
    withIndex(0);
  }
};

struct test_build_insertion_progress : public vrp::algorithms::construction::build_insertion_progress {
  explicit test_build_insertion_progress() : vrp::algorithms::construction::build_insertion_progress() {
    withCost(10).withCompleteness(1);
  }
};
}
