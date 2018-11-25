#pragma once

#include "algorithms/construction/insertion/InsertionActivityContext.hpp"
#include "algorithms/construction/insertion/InsertionContext.hpp"
#include "algorithms/construction/insertion/InsertionProgress.hpp"
#include "algorithms/construction/insertion/InsertionRouteContext.hpp"

#include <limits>
#include <memory>

namespace vrp::algorithms::construction {

/// Builds insertion progress.
class build_insertion_progress {
public:
  build_insertion_progress& cost(models::common::Cost value) {
    progress_.bestCost = value;
    return *this;
  }

  build_insertion_progress& completeness(double value) {
    progress_.completeness = value;
    return *this;
  }

  InsertionProgress&& owned() { return std::move(progress_); }

  std::shared_ptr<InsertionProgress> shared() { return std::make_shared<InsertionProgress>(std::move(progress_)); }

private:
  InsertionProgress progress_;
};

/// Creates build insertion context.
class build_insertion_context {
public:
  explicit build_insertion_context() : context_({{std::numeric_limits<models::common::Cost>::max(), 0}, {}, {}}) {}

  build_insertion_context& progress(const InsertionProgress& value) {
    context_.progress = value;
    return *this;
  }

  build_insertion_context& jobs(std::vector<models::problem::Job>&& value) {
    context_.jobs = value;
    return *this;
  }

  build_insertion_context& routes(std::vector<InsertionContext::RouteState>&& value) {
    context_.routes = value;
    return *this;
  }

  InsertionContext&& owned() { return std::move(context_); }

  std::shared_ptr<InsertionContext> shared() { return std::make_shared<InsertionContext>(std::move(context_)); }

private:
  InsertionContext context_;
};

/// Creates insertion route context.
class build_insertion_route_context {
public:
  build_insertion_route_context& actor(models::solution::Route::Actor value) {
    context_.actor = std::move(value);
    return *this;
  }

  build_insertion_route_context& route(std::shared_ptr<models::solution::Route> value) {
    context_.route = std::move(value);
    return *this;
  }

  build_insertion_route_context& departure(models::common::Timestamp departure) {
    context_.departure = departure;
    return *this;
  }

  build_insertion_route_context& state(std::shared_ptr<InsertionRouteState> value) {
    context_.state = std::move(value);
    return *this;
  }

  InsertionRouteContext&& owned() { return std::move(context_); }

  std::shared_ptr<InsertionRouteContext> shared() {
    return std::make_shared<InsertionRouteContext>(std::move(context_));
  }

protected:
  InsertionRouteContext context_;
};

/// Creates insertion activity context.
class build_insertion_activity_context {
public:
  build_insertion_activity_context& index(size_t value) {
    context_.index = value;
    return *this;
  }

  build_insertion_activity_context& departure(models::common::Timestamp value) {
    context_.departure = value;
    return *this;
  }

  build_insertion_activity_context& prev(models::solution::Tour::Activity value) {
    context_.prev = std::move(value);
    return *this;
  }

  build_insertion_activity_context& target(models::solution::Tour::Activity value) {
    context_.target = std::move(value);
    return *this;
  }

  build_insertion_activity_context& next(models::solution::Tour::Activity value) {
    context_.next = std::move(value);
    return *this;
  }

  InsertionActivityContext&& owned() { return std::move(context_); }

  std::shared_ptr<InsertionActivityContext> shared() {
    return std::make_shared<InsertionActivityContext>(std::move(context_));
  }

private:
  InsertionActivityContext context_;
};
}
