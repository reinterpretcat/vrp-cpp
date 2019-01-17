#pragma once

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"
#include "test_utils/models/Extensions.hpp"

#include <catch/catch.hpp>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>

namespace vrp::test {

/// Validates solution against time and size violations.
/// NOTE supports only deliveries at the moment.
template<typename Size, bool AllowUnassigned>
struct validate_solution final {
  using SizeHandler = typename algorithms::construction::VehicleActivitySize<Size>;

  /// Keeps current state.
  struct State final {
    std::shared_ptr<const models::solution::Route> route = {};
    models::common::Location location;
    models::common::Timestamp time = {};
    Size size = {};
  };

  void operator()(const models::Problem& problem, const models::Solution& solution) const {
    checkIds(problem, solution);

    ranges::for_each(solution.routes, [&](const auto& route) {
      auto state = State{route, route->start->detail.location, route->start->schedule.departure, Size{}};
      checkActivity(problem, state, route->start);

      ranges::for_each(route->tour.activities(),
                       [&](const auto& activity) { checkActivity(problem, state, activity); });

      checkActivity(problem, state, route->end);
    });
  }

private:
  void checkIds(const models::Problem& problem, const models::Solution& solution) const {
    auto ids =
      solution.routes | ranges::view::for_each([](const auto& r) {
        return r->tour.activities() | ranges::view::transform([](const auto& a) { return get_job_id{}(*a->job); });
      }) |
      ranges::to_vector | ranges::action::sort;

    if (ids.size() + (AllowUnassigned ? solution.unassigned.size() : 0) != problem.jobs->size())
      fail("unexpected job ids");
  }

  void checkActivity(const models::Problem& problem,
                     State& state,
                     const models::solution::Tour::Activity& activity) const {
    // logActivity(activity);
    checkSize(state, activity);
    checkTime(problem, state, activity);
    checkLocation(state, activity);
    // logState(state);
  }

  void checkLocation(State& state, const models::solution::Tour::Activity& activity) const {
    state.location = activity->detail.location;
  }

  void checkSize(State& state, const models::solution::Tour::Activity& activity) const {
    auto size = SizeHandler::getSize(activity);
    state.size += size;
    if (state.size > SizeHandler::getSize(state.route->actor->vehicle)) fail("size is exceeded");
  }

  void checkTime(const models::Problem& problem, State& state, const models::solution::Tour::Activity& activity) const {
    auto driving = problem.transport->duration(
      state.route->actor->vehicle->profile, state.location, activity->detail.location, state.time);

    auto arrival = state.time + driving;
    if (arrival > activity->detail.time.end) fail("arrival after tw end time");
    if (arrival != activity->schedule.arrival) fail("wrong arrival");

    auto waiting = arrival > activity->detail.time.start ? 0 : activity->detail.time.start - arrival;

    auto serviceStart = arrival + waiting;
    auto departure = serviceStart + problem.activity->duration(*state.route->actor, *activity, serviceStart);
    if (departure != activity->schedule.departure) fail("wrong departure");

    state.time = departure;
  }

  void check(const Size& current, const Size& limit) {
    if (current > limit) fail("wrong size");
  }

  void logActivity(const models::solution::Tour::Activity& activity) const {
    std::cout << "activity:: "
              << "job:" << getId(activity) << ", schedule: [" << activity->schedule.arrival << ","
              << activity->schedule.departure << "], time: [" << activity->detail.time.start << ","
              << activity->detail.time.end << "], location:" << activity->detail.location
              << ", duration: " << activity->detail.duration << ", size:" << SizeHandler::getSize(activity)
              << std::endl;
  }

  void logState(const State& state) const {
    std::cout << "state::    "
              << "route:" << state.route->actor->vehicle->id << ", location:" << state.location
              << ", timestamp:" << state.time << ", size:" << state.size << std::endl;
  }

  std::string getId(const models::solution::Tour::Activity& activity) const {
    if (activity->type == models::solution::Activity::Type::Start) return "start";
    if (activity->type == models::solution::Activity::Type::End) return "end";
    return get_job_id{}(*activity->job);
  }

  void fail(const std::string& msg) const {
    //
    REQUIRE("error occurred:" == msg);
  }
};
}