#include "streams/output/GeoJsonWriter.hpp"

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <json/json11.hpp>

#include <algorithm>
#include <ostream>

using namespace json11;
using namespace vrp::models;
using namespace vrp::streams;

using namespace std::placeholders;

namespace {

/// Represents a job defined by customer and vehicle.
using Job = thrust::tuple<int,int>;
/// Represents tours defined by tour's vehicle and its customers.
using Tours = std::map<int, std::vector<int>>;
/// Represents location resolver.
using LocationResolver = GeoJsonWriter::LocationResolver;

/// Creates coordinate from customer id.
Json::array createCoordinate(const LocationResolver &resolver, int id) {
  auto coord = resolver(id);
  return Json::array { coord.first, coord.second };
}

/// Creates geo json with line string for given tour.
Json createLineString(const LocationResolver &resolver, Tours &tours, int solution, int tour) {
  return Json::object {
      {"type", "Feature"},
      {"properties", Json::object{
          {"solution", solution},
          {"tour", tour}
      }},
      {"geometry", Json::object {
          {"type", "LineString"},
          {"coordinates", thrust::transform_reduce(
              thrust::host,
              tours[tour].begin(),
              tours[tour].end(),
              [&](int id) { return createCoordinate(resolver, id); },
              Json::array {createCoordinate(resolver, 0), createCoordinate(resolver, 0)},
              [](Json::array &result, const Json::array &item) {
                result.insert(result.end() - 1, item);
                return result;
              }
          )}
      }}
  };
}

Json mergeTours(Json &result, const Json &tour) {
  auto obj = result.object_items();
  auto features = obj["features"].array_items();
  features.push_back(tour);
  obj["features"] = features;
  return obj;
}

Json mergeSolutions(Json &result, const Json &solution) {
  auto obj = result.object_items();
  auto features = solution["features"].array_items();
  obj["features"] = features;
  return obj;
}

/// Gets tours created from sorted jobs.
Tours getTours(const std::vector<Job> jobs) {
  auto tours = Tours();
  std::for_each(
      jobs.begin(),
      jobs.end(),
      [&](const Job &job) {
        tours[thrust::get<1>(job)].push_back(thrust::get<0>(job));
      });

  return tours;
}

/// Materializes single solution as geo json.
struct materialize_solution final {
  const Tasks &tasks;
  const LocationResolver &resolver;

  json11::Json operator()(int solution) {
    auto jobs = getJobs(solution);
    auto tours = getTours(std::vector<Job>(jobs.begin(), jobs.end()));

    return thrust::transform_reduce(
        thrust::host,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(static_cast<int>(tours.size())),
        [&](int tour) { return createLineString(resolver, tours, solution, tour); },
        Json(),
        std::bind(&mergeTours, _1, _2));
  }

  __host__ __device__
  Job operator()(const Job &item) { return item; }

  __host__ __device__
  int operator()(const Job &left, const Job &right) {
    return thrust::get<0>(left) < thrust::get<0>(right);
  }

  /// Gets all jobs in sorted by vehicle order.
  thrust::host_vector<Job> getJobs(int solution) {
    int start = tasks.customers * solution;
    int end = start + tasks.customers;
    thrust::device_vector<Job> jobs(static_cast<std::size_t>(tasks.customers - 1));

    // get all jobs with their vehicles (except depot)
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            tasks.ids.data() + start + 1,
            tasks.vehicles.data() + start)),
        thrust::make_zip_iterator(thrust::make_tuple(
            tasks.ids.data() + end,
            tasks.vehicles.data() + end)),
        jobs.begin(),
        *this
    );

    return jobs;
  }
};
}

namespace vrp {
namespace streams {

void GeoJsonWriter::write(std::ostream &out,
                          const vrp::models::Tasks &tasks,
                          const LocationResolver &resolver) {

  int solutions = static_cast<int>(tasks.ids.size() / tasks.customers);
  Json json = Json::object {
      {"type", "FeatureCollection"},
      {"features", Json::array()}
  };
  out <<  thrust::transform_reduce(
      thrust::host,
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(solutions),
      materialize_solution{tasks, resolver},
      json,
      std::bind(&mergeSolutions, _1, _2)).dump();
}

}
}
