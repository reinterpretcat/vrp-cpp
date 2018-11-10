#pragma once

#include "algorithms/construction/extensions/Insertions.hpp"
#include "test_utils/models/Factories.hpp"

namespace vrp::test {
struct test_build_insertion_context : public vrp::algorithms::construction::build_insertion_context {
  explicit test_build_insertion_context() : vrp::algorithms::construction::build_insertion_context() {
    withActor(DefaultActor).withRoute(test_build_route{}.shared()).withTime(DefaultTime);
  }
};
}