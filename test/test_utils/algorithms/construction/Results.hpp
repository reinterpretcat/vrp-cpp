#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "models/extensions/problem/Helpers.hpp"
#include "models/extensions/solution/Helpers.hpp"

#include <map>
#include <range/v3/all.hpp>
#include <set>
#include <vector>

namespace vrp::test {

/// Returns vector of job ids from all routes.
struct get_job_ids_from_all_routes {
  std::vector<std::string> operator()(const models::Solution& solution) const {
    return operator()(solution.routes, [](const auto r) { return r; });
  }

  std::vector<std::string> operator()(const algorithms::construction::InsertionContext& ctx) const {
    return operator()(ctx.solution->routes, [](const auto r) { return r.route; });
  }

  std::vector<std::string> operator()(std::shared_ptr<const models::solution::Route>& route) const {
    return operator()(ranges::view::single(route), [](const auto r) { return r; });
  }

private:
  template<typename Routes, typename MapRoute>
  std::vector<std::string> operator()(const Routes& routes, MapRoute map) const {
    using namespace ranges;

    return routes | view::for_each([&](const auto& r) {
             return map(r)->tour.activities() | view::transform([](const auto& a) {
                      auto job = models::solution::retrieve_job{}(*a);
                      return job.has_value() ? models::problem::get_job_id{}(job.value()) : "";
                    });
           }) |
      ranges::view::remove_if([](const auto& id) { return id == ""; }) | to_vector;
  }
};

/// Returns vector of service ids from all routes.
struct get_service_ids_from_all_routes {
  std::vector<std::string> operator()(const algorithms::construction::InsertionContext& ctx) const {
    using namespace ranges;

    return ctx.solution->routes | view::for_each([](const auto& r) {
             return r.route->tour.activities() |
               view::remove_if([](const auto& a) { return !a->service.has_value(); }) |
               view::transform([](const auto& a) { return models::problem::getId(a->service.value()->dimens); });
           }) |
      to_vector;
  }
};

/// Returns vectors of job ids from each route separately.
struct get_job_ids_from_routes final {
  std::vector<std::vector<std::string>> operator()(const algorithms::construction::InsertionContext& ctx) const {
    using namespace ranges;

    return ctx.solution->routes | view::transform([](const auto& r) {
             return r.route->tour.activities() | view::transform([](const auto& a) {
                      return vrp::models::problem::get_job_id{}(models::solution::retrieve_job{}(*a).value());
                    }) |
               to_vector;
           }) |
      to_vector;
  }
};

/// Returns sorted job ids from jobs.
struct get_job_ids_from_jobs final {
  std::vector<std::string> operator()(const std::vector<models::problem::Job>& jobs) const {
    using namespace ranges;

    return view::all(jobs) | view::transform([](const auto& j) { return vrp::models::problem::get_job_id{}(j); }) |
      to_vector | action::sort;
  }

  std::vector<std::string> operator()(
    const std::map<models::problem::Job, int, models::problem::compare_jobs>& jobs) const {
    using namespace ranges;

    return view::all(jobs) |
      view::transform([](const auto& pair) { return vrp::models::problem::get_job_id{}(pair.first); }) | to_vector |
      action::sort;
  }

  std::vector<std::string> operator()(const std::set<models::problem::Job, models::problem::compare_jobs>& jobs) const {
    using namespace ranges;

    return jobs | view::transform([](const auto& j) { return models::problem::get_job_id{}(j); }) | to_vector |
      action::sort;
  }
};
}