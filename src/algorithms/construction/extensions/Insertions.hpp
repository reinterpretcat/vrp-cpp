#pragma once

#include "algorithms/construction/insertion/InsertionContext.hpp"

namespace vrp::algorithms::construction {

class build_insertion_context {
public:
  build_insertion_context& withActor(std::shared_ptr<models::problem::Actor> actor) {
    context_.actor = actor;
    return *this;
  }

  build_insertion_context& withRoute(std::shared_ptr<models::solution::Route> route) {
    context_.route = route;
    return *this;
  }

  build_insertion_context& withTime(models::common::Timestamp time) {
    context_.time = time;
    return *this;
  }

  InsertionContext&& owned() { return std::move(context_); }

private:
  InsertionContext context_;
};
}
