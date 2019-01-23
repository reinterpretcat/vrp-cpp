#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "models/extensions/problem/Properties.hpp"

#include <range/v3/all.hpp>

namespace vrp::test {

/// Returns vector of job ids from all routes.
struct get_job_ids_from_all_routes {
  std::vector<std::string> operator()(const algorithms::construction::InsertionContext& ctx) const {
    using namespace ranges;

    return ctx.routes | view::for_each([](const auto& r) {
             return r.route->tour.activities() |
               view::transform([](const auto& a) { return vrp::models::problem::get_job_id{}(*a->job); });
           }) |
      to_vector;
  }
};

/// Returns vectors of job ids from each route separately.
struct get_job_ids_from_routes final {
  std::vector<std::vector<std::string>> operator()(const algorithms::construction::InsertionContext& ctx) const {
    using namespace ranges;

    return ctx.routes | view::transform([](const auto& r) {
             return r.route->tour.activities() |
               view::transform([](const auto& a) { return vrp::models::problem::get_job_id{}(*a->job); }) | to_vector;
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
};
}