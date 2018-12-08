#pragma once

#include "algorithms/construction/InsertionContext.hpp"
#include "test_utils/models/Extensions.hpp"

#include <range/v3/all.hpp>

namespace vrp::test {

/// Returns vector of job ids from all routes.
struct get_job_ids_from_all_routes {
  std::vector<std::string> operator()(const algorithms::construction::InsertionContext& ctx) {
    using namespace ranges;

    return ctx.routes | view::for_each([](const auto& r) {
             return r.first->tour.activities() |
               view::transform([](const auto& a) { return vrp::test::get_job_id{}(*a->job); });
           }) |
      to_vector;
  }
};

/// Returns vectors of job ids from each route separately.
struct get_job_ids_from_routes final {
  std::vector<std::vector<std::string>> operator()(const algorithms::construction::InsertionContext& ctx) {
    using namespace ranges;

    return ctx.routes | view::transform([](const auto& r) {
             return r.first->tour.activities() |
               view::transform([](const auto& a) { return vrp::test::get_job_id{}(*a->job); }) | to_vector;
           }) |
      to_vector;
  }
};
}