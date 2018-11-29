#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "test_utils/models/Factories.hpp"

namespace vrp::test {

struct test_build_insertion_progress : public vrp::algorithms::construction::build_insertion_progress {
  explicit test_build_insertion_progress() : vrp::algorithms::construction::build_insertion_progress() {
    cost(1000).completeness(1);
  }
};

struct test_build_insertion_context : public vrp::algorithms::construction::build_insertion_context {
  explicit test_build_insertion_context() : vrp::algorithms::construction::build_insertion_context() {
    progress(test_build_insertion_progress{}.owned());
  }
};

struct test_build_insertion_route_context : public vrp::algorithms::construction::build_insertion_route_context {
  explicit test_build_insertion_route_context() : vrp::algorithms::construction::build_insertion_route_context() {
    actor(DefaultActor)
      .route({test_build_route{}.shared(), std::make_shared<algorithms::construction::InsertionRouteState>()})
      .departure(DefaultTime);
  }

  test_build_insertion_route_context& add(const models::solution::Tour::Activity& activity) {
    context_.route.first->tour.add(activity);
    return *this;
  }
};

struct test_build_insertion_activity_context : public vrp::algorithms::construction::build_insertion_activity_context {
  explicit test_build_insertion_activity_context() : vrp::algorithms::construction::build_insertion_activity_context() {
    index(0);
  }
};
}
