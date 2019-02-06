#pragma once

#include "algorithms/construction/InsertionActivityContext.hpp"
#include "algorithms/construction/InsertionContext.hpp"
#include "algorithms/construction/InsertionProgress.hpp"
#include "algorithms/construction/InsertionRouteContext.hpp"
#include "models/extensions/problem/Comparators.hpp"
#include "models/extensions/solution/DeepCopies.hpp"

#include <cmath>
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
    progress_.completeness = std::max(0.5, value);
    return *this;
  }

  build_insertion_progress& total(int value) {
    progress_.total = value;
    return *this;
  }

  InsertionProgress&& owned() {
    assert(progress_.total > 0);
    return std::move(progress_);
  }

  std::shared_ptr<InsertionProgress> shared() {
    assert(progress_.total > 0);
    return std::make_shared<InsertionProgress>(std::move(progress_));
  }

private:
  InsertionProgress progress_;
};

/// Creates build insertion context.
class build_insertion_context {
public:
  explicit build_insertion_context() : context_({{std::numeric_limits<models::common::Cost>::max(), 0}, {}, {}, {}}) {}

  build_insertion_context& problem(const std::shared_ptr<const models::Problem>& value) {
    context_.problem = value;
    return *this;
  }

  build_insertion_context& registry(const std::shared_ptr<models::solution::Registry>& value) {
    context_.registry = value;
    return *this;
  }

  build_insertion_context& progress(const InsertionProgress& value) {
    context_.progress = value;
    return *this;
  }

  build_insertion_context& jobs(std::vector<models::problem::Job>&& value) {
    context_.jobs = std::move(value);
    return *this;
  }

  build_insertion_context& routes(std::set<InsertionRouteContext, compare_insertion_route_contexts>&& value) {
    context_.routes = std::move(value);
    return *this;
  }

  build_insertion_context& random(std::shared_ptr<utils::Random> random) {
    context_.random = random;
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
  build_insertion_route_context& route(const std::shared_ptr<models::solution::Route>& value) {
    context_.route = value;
    return *this;
  }

  build_insertion_route_context& state(const std::shared_ptr<InsertionRouteState>& value) {
    context_.state = value;
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

/// Creates a deep copy of insertion route context.
struct deep_copy_insertion_route_context final {
  InsertionRouteContext operator()(const InsertionRouteContext& rs) const {
    using namespace ranges;
    using namespace vrp::models::solution;

    auto state = std::make_shared<InsertionRouteState>(rs.state->sizes());
    auto route = std::make_shared<Route>();

    // copy tour and activity level states
    auto tour = view::concat(view::single(rs.route->start), rs.route->tour.activities(), view::single(rs.route->end));
    ranges::for_each(tour, [&](const auto& a) {
      auto clone = std::make_shared<Activity>(Activity{*a});

      ranges::for_each(rs.state->keys(), [&](const auto& key) {
        auto aValue = rs.state->get(key, a);
        if (aValue) state->put(key, clone, aValue.value());
      });

      if (a->service.has_value()) {
        route->tour.add(clone);
      } else {
        if (route->start)
          route->end = clone;
        else
          route->start = clone;
      }
    });

    // copy tour level states
    ranges::for_each(rs.state->keys(), [&](const auto& key) {
      auto rValue = rs.state->get(key);
      if (rValue) state->put(key, rValue.value());
    });

    // copy actor
    route->actor = rs.route->actor;

    return {route, state};
  }
};
}
