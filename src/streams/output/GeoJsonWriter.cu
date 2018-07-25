#include "runtime/Config.hpp"
#include "streams/output/GeoJsonWriter.hpp"

#include <algorithm>
#include <json/json11.hpp>
#include <ostream>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

using namespace json11;
using namespace vrp::models;
using namespace vrp::runtime;
using namespace vrp::streams;

using namespace std::placeholders;

namespace {

/// Represents a job defined by customer and vehicle.
using Job = thrust::tuple<int, int>;
/// Represents a single tour.
using Tour = std::vector<int>;
/// Represents tours defined by tour's vehicle and its customers.
using Tours = std::map<int, Tour>;
/// Represents location resolver.
using LocationResolver = GeoJsonWriter::LocationResolver;

/// Creates coordinate from customer id.
Json::array createCoordinate(const LocationResolver& resolver, int id) {
  auto coord = resolver(id);
  return Json::array{coord.first, coord.second};
}

/// Creates geo json of given route as line string.
Json createRoute(const LocationResolver& resolver, int solutionId, int tourId, Tour& tour) {
  return Json::object{
    {"type", "Feature"},
    {"properties", Json::object{{"solution", solutionId}, {"tour", tourId}}},
    {"geometry",
     Json::object{
       {"type", "LineString"},
       {"coordinates", thrust::transform_reduce(
                         thrust::host, tour.begin(), tour.end(),
                         [&](int id) { return createCoordinate(resolver, id); },
                         Json::array{createCoordinate(resolver, 0), createCoordinate(resolver, 0)},
                         [](Json::array& result, const Json::array& item) {
                           result.insert(result.end() - 1, item);
                           return result;
                         })}}}};
}

/// Creates geojson of given job as point.
Json createJob(const LocationResolver& resolver, int solutionId, int vehicleId, int customerId) {
  return Json::object{
    {"type", "Feature"},
    {"properties",
     Json::object{
       {"solution", solutionId},
       {"customer", customerId},
       {"vehicle", vehicleId},
     }},
    {"geometry",
     Json::object{{"type", "Point"}, {"coordinates", createCoordinate(resolver, customerId)}}}};
}

/// Creates tour as collection of geojson features.
Json createTour(const LocationResolver& resolver, int solutionId, int tourId, Tour& tour) {
  Json::array features{createRoute(resolver, solutionId, tourId, tour)};

  std::transform(tour.begin(), tour.end(), std::back_inserter(features), [&](int customerId) {
    return createJob(resolver, solutionId, tourId, customerId);
  });

  return features;
};

/// Merges tour into result.
Json mergeTour(Json& result, const Json& tour) {
  auto obj = result.object_items();
  auto features = obj["features"].array_items();
  auto tourFeatures = tour.array_items();

  features.insert(features.end(), tourFeatures.begin(), tourFeatures.end());

  obj["features"] = features;
  return obj;
}

/// Merges solution into result.
Json mergeSolution(Json& result, const Json& solution) {
  auto obj = result.object_items();
  auto features = solution["features"].array_items();
  obj["features"] = features;
  return obj;
}

/// Gets tours created from sorted jobs.
Tours getTours(const std::vector<Job> jobs) {
  auto tours = Tours();
  std::for_each(jobs.begin(), jobs.end(),
                [&](const Job& job) { tours[thrust::get<1>(job)].push_back(thrust::get<0>(job)); });

  return tours;
}

/// Materializes single solution as geo json.
struct materialize_solution final {
  const Tasks& tasks;
  const LocationResolver& resolver;

  json11::Json operator()(int solution) {
    auto jobs = getJobs(solution);
    auto tours = getTours(std::vector<Job>(jobs.begin(), jobs.end()));

    return thrust::transform_reduce(
      thrust::host, thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(static_cast<int>(tours.size())),
      [&](int tour) { return createTour(resolver, solution, tour, tours[tour]); }, Json(),
      std::bind(&mergeTour, _1, _2));
  }

  ANY_EXEC_UNIT Job operator()(const Job& item) { return item; }

  /// Gets all jobs in sorted by vehicle order.
  thrust::host_vector<Job> getJobs(int solution) {
    int start = tasks.customers * solution;
    int end = start + tasks.customers;
    vector<Job> jobs(static_cast<std::size_t>(tasks.customers - 1));

    // get all jobs with their vehicles (except depot)
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(tasks.ids.data() + start + 1,
                                                                   tasks.vehicles.data() + start)),
                      thrust::make_zip_iterator(
                        thrust::make_tuple(tasks.ids.data() + end, tasks.vehicles.data() + end)),
                      jobs.begin(), *this);

    return jobs;
  }
};
}  // namespace

namespace vrp {
namespace streams {

void GeoJsonWriter::write(std::ostream& out,
                          const vrp::models::Tasks& tasks,
                          const LocationResolver& resolver) {
  int solutions = static_cast<int>(tasks.ids.size() / tasks.customers);
  Json json = Json::object{{"type", "FeatureCollection"}, {"features", Json::array()}};
  out << thrust::transform_reduce(thrust::host, thrust::counting_iterator<int>(0),
                                  thrust::counting_iterator<int>(solutions),
                                  materialize_solution{tasks, resolver}, json,
                                  std::bind(&mergeSolution, _1, _2))
           .dump();
}

}  // namespace streams
}  // namespace vrp
