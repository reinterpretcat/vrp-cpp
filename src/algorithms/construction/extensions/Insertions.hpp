#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"

#include <memory>

namespace vrp::algorithms::construction {

/// Creates insertion route context.
class build_insertion_route_context {
public:
  build_insertion_route_context& withActor(std::shared_ptr<models::problem::Actor> actor) {
    context_.actor = std::move(actor);
    return *this;
  }

  build_insertion_route_context& withRoute(std::shared_ptr<models::solution::Route> route) {
    context_.route = std::move(route);
    return *this;
  }

  build_insertion_route_context& withTime(models::common::Timestamp time) {
    context_.time = time;
    return *this;
  }

  build_insertion_route_context& withState(std::shared_ptr<InsertionRouteState> state) {
    context_.state = std::move(state);
    return *this;
  }

  InsertionRouteContext&& owned() { return std::move(context_); }

  std::shared_ptr<InsertionRouteContext> shared() {
    return std::make_shared<InsertionRouteContext>(std::move(context_));
  }

private:
  InsertionRouteContext context_;
};

/// Creates insertion activity context.
class build_insertion_activity_context {
public:
  build_insertion_activity_context& withIndex(size_t index) {
    context_.index = index;
    return *this;
  }

  build_insertion_activity_context& withPrev(models::solution::Tour::Activity prev) {
    context_.prev = std::move(prev);
    return *this;
  }

  build_insertion_activity_context& withTarget(models::solution::Tour::Activity target) {
    context_.target = std::move(target);
    return *this;
  }

  build_insertion_activity_context& withNext(models::solution::Tour::Activity next) {
    context_.next = std::move(next);
    return *this;
  }

  InsertionActivityContext&& owned() { return std::move(context_); }

  std::shared_ptr<InsertionActivityContext> shared() {
    return std::make_shared<InsertionActivityContext>(std::move(context_));
  }

private:
  InsertionActivityContext context_;
};

class build_insertion_progress {
public:
  build_insertion_progress& withCost(models::common::Cost cost) {
    progress_.bestCost = cost;
    return *this;
  }

  build_insertion_progress& withCompleteness(double completeness) {
    progress_.completeness = completeness;
    return *this;
  }

  InsertionProgress&& owned() { return std::move(progress_); }

  std::shared_ptr<InsertionProgress> shared() { return std::make_shared<InsertionProgress>(std::move(progress_)); }

private:
  InsertionProgress progress_;
};
}
