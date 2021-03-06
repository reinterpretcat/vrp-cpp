#pragma once

#include "algorithms/construction/constraints/VehicleActivitySize.hpp"
#include "models/Problem.hpp"
#include "models/Solution.hpp"

#include <catch/catch.hpp>
#include <iostream>
#include <optional>
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
    using namespace ranges;

    checkIds(problem, solution);

    ranges::for_each(solution.routes, [&](const auto& route) {
      auto state = State{route, route->tour.start()->detail.location, route->tour.start()->schedule.arrival, Size{}};

      ranges::for_each(view::zip(route->tour.activities(), view::iota(0)), [&](const auto& indexedActivity) {
        auto [activity, index] = indexedActivity;
        auto next = index == 0 ? std::make_optional<models::solution::Tour::Activity>(route->tour.get(1))
                               : std::make_optional<models::solution::Tour::Activity>();
        checkActivity(problem, state, index, activity, next);
      });
    });
  }

private:
  void checkIds(const models::Problem& problem, const models::Solution& solution) const {
    using namespace models::problem;
    using namespace ranges;
    auto ids = solution.routes | view::for_each([](const auto& r) {
                 return r->tour.activities() |  //
                   view::remove_if([](const auto& a) { return !a->service.has_value(); }) |
                   view::transform([](const auto& a) { return get_job_id{}(as_job(a->service.value())); });
               }) |
      to_vector | action::sort;

    auto actual = ids.size() + (AllowUnassigned ? solution.unassigned.size() : 0);

    auto expected = ranges::accumulate(problem.jobs->all(), 0, [](auto acc, const auto& job) {
      return acc +
        models::problem::analyze_job<size_t>(
               job,
               [&](const std::shared_ptr<const models::problem::Service>& service) { return 1; },
               [&](const std::shared_ptr<const models::problem::Sequence>& sequence) {
                 return sequence->services.size();
               });
    });

    if (actual != expected) fail("unexpected job ids: " + std::to_string(actual) + " vs " + std::to_string(expected));
  }

  void checkActivity(const models::Problem& problem,
                     State& state,
                     int index,
                     const models::solution::Tour::Activity& activity,
                     const std::optional<models::solution::Tour::Activity>& next) const {
    // logActivity(activity);
    checkSize(state, activity);
    checkTime(problem, state, index, activity, next);
    checkLocation(state, activity);
    // logState(state);
  }

  void checkLocation(State& state, const models::solution::Tour::Activity& activity) const {
    state.location = activity->detail.location;
  }

  void checkSize(State& state, const models::solution::Tour::Activity& activity) const {
    auto size = SizeHandler::getDemand(activity);
    // TODO support services
    state.size += size.delivery.first;
    if (state.size > SizeHandler::getCapacity(state.route->actor->vehicle)) fail("size is exceeded");
  }

  void checkTime(const models::Problem& problem,
                 State& state,
                 int index,
                 const models::solution::Tour::Activity& activity,
                 const std::optional<models::solution::Tour::Activity>& next) const {
    auto driving = problem.transport->duration(
      state.route->actor->vehicle->profile, state.location, activity->detail.location, state.time);

    auto arrival = state.time + driving;

    if (arrival > activity->detail.time.end)
      fail(std::to_string(activity->detail.location) + " has arrival after tw end time");

    if (arrival != activity->schedule.arrival) fail(std::to_string(activity->detail.location) + " has wrong arrival");

    auto waiting = arrival > activity->detail.time.start ? 0 : activity->detail.time.start - arrival;

    auto serviceStart = arrival + waiting;
    auto departure = serviceStart + problem.activity->duration(*state.route->actor, *activity, serviceStart);

    if (index == 0) {
      if (!next) fail("tour has only departure");
      departure = next.value()->schedule.arrival -
        problem.transport->duration(
          state.route->actor->vehicle->profile, activity->detail.location, next.value()->detail.location, state.time);
    }

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
              << ", duration: " << activity->detail.duration << ", demand:" << getDemandString(activity) << std::endl;
  }

  void logState(const State& state) const {
    std::cout << "state::    "
              << "route:" << state.route->actor->vehicle->id << ", location:" << state.location
              << ", timestamp:" << state.time << ", size:" << state.size << std::endl;
  }

  std::string getId(const models::solution::Tour::Activity& activity) const {
    return activity->service.has_value()
      ? models::problem::get_job_id{}(models::solution::retrieve_job{}(*activity).value())
      : "|";
  }

  std::string getDemandString(const models::solution::Tour::Activity& activity) const {
    auto demand = SizeHandler::getDemand(activity);
    return "{TODO}";
  }

  void fail(const std::string& msg) const {
    //
    REQUIRE("error occurred:" == msg);
  }
};
}