#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "test_utils/models/Factories.hpp"

namespace vrp::test {

struct test_build_insertion_progress : public vrp::algorithms::construction::build_insertion_progress {
  explicit test_build_insertion_progress() : vrp::algorithms::construction::build_insertion_progress() {
    cost(1000).completeness(1).total(1);
  }
};

struct test_build_insertion_context : public vrp::algorithms::construction::build_insertion_context {
  explicit test_build_insertion_context() : vrp::algorithms::construction::build_insertion_context() {
    progress(test_build_insertion_progress{}.owned()).random(std::make_shared<utils::Random>());
  }
};

struct test_build_insertion_route_context : public vrp::algorithms::construction::build_insertion_route_context {
  explicit test_build_insertion_route_context() : vrp::algorithms::construction::build_insertion_route_context() {
    route(test_build_route{}.actor(DefaultActor).shared())
      .state(std::make_shared<algorithms::construction::InsertionRouteState>());
  }

  test_build_insertion_route_context& insert(const models::solution::Tour::Activity& activity, size_t index) {
    context_.route->tour.insert(activity, index);
    return *this;
  }
};

struct test_build_insertion_activity_context : public vrp::algorithms::construction::build_insertion_activity_context {
  explicit test_build_insertion_activity_context() : vrp::algorithms::construction::build_insertion_activity_context() {
    index(0);
  }
};
}
