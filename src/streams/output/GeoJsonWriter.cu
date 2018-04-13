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

namespace {
/// Materializes single solution as geo json.
struct materialize_solution final {
  /// Represents a job defined by customer and vehicle.
  using Job = thrust::tuple<int,int>;
  /// Represents tours defined by tour's vehicle and its customers.
  using Tours = std::map<int, std::vector<int>>;

  const Tasks &tasks;
  const GeoJsonWriter::LocationResolver &resolver;

  json11::Json operator()(int solution) {
    auto jobs = getJobs(solution);
    auto tours = getTours(std::vector<Job>(jobs.begin(), jobs.end()));

    auto toCoordinate = [&](int id) {
      auto coord = resolver(id);
      return Json::array { coord.first, coord.second };
    };

    return thrust::transform_reduce(
        thrust::host,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(static_cast<int>(tours.size())),
        [&](int tour) {

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
                      toCoordinate,
                      Json::array {toCoordinate(0), toCoordinate(0)},
                      [](Json::array &result, const Json::array &item) {
                        result.insert(result.end() - 1, item);
                        return result;
                      }
                  )}
              }}
          };
        },
        Json(),
        [](Json &result, const Json &tour) {
          auto obj = result.object_items();
          auto features = obj["features"].array_items();
          features.push_back(tour);
          obj["features"] = features;
          return obj;
        });
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

  __host__ __device__
  Job operator()(const Job &item) { return item; }

  __host__ __device__
  int operator()(const Job &left, const Job &right) {
    return thrust::get<0>(left) < thrust::get<0>(right);
  }
};

/// Merges solution into json result.
struct merge_solution final {
  Json operator()(Json &result, const Json &solution) {
    auto obj = result.object_items();
    auto features = solution["features"].array_items();
    obj["features"] = features;
    return obj;
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
      merge_solution()).dump();
}

}
}
