#ifndef VRP_STREAMS_GEOJSONWRITER_HPP
#define VRP_STREAMS_GEOJSONWRITER_HPP

#include "models/Tasks.hpp"
#include <json/json11.hpp>

#include <algorithm>
#include <ostream>

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace vrp {
namespace streams {

/// Writes solution into output stream  in geojson format.
template <typename LocationResolver>
class GeoJsonWriter final {
 public:
  /// Materializes single solution as geo json.
  struct materialize_solution final {
    /// Represents a job defined by customer and vehicle.
    using Job = thrust::tuple<int,int>;
    /// Represents tours defined by tour's vehicle and its customers.
    using Tours = std::map<int, std::vector<int>>;

    const vrp::models::Tasks &tasks;
    const LocationResolver &resolver;

    json11::Json operator()(int solution) {
      auto jobs = getJobs(solution);
      auto tours = getTours(std::vector<Job>(jobs.begin(), jobs.end()));

      return thrust::transform_reduce(
          thrust::host,
          thrust::counting_iterator<int>(0),
          thrust::counting_iterator<int>(static_cast<int>(tours.size())),
          [&](int tour) {
            auto toCoordinate = [&](int id) {
              auto coord = resolver(id);
              return json11::Json::array { coord.first, coord.second };
            };
            return json11::Json::object {
              {"solution", solution},
              {"tour", tour},
              {"geometry", json11::Json::object {
                  {"type", "LineString"},
                  {"coordinates", thrust::transform_reduce(
                      thrust::host,
                      tours[tour].begin(),
                      tours[tour].end(),
                      toCoordinate,
                      json11::Json::array {toCoordinate(0), toCoordinate(0)},
                      [](json11::Json::array &result, const json11::Json::array &item) {
                        result.insert(result.end() - 1, item);
                        return result;
                      }
                  )}
              }}
            };
          },
          json11::Json(),
          [](json11::Json &result, const json11::Json &tour) {
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
    json11::Json operator()(json11::Json &result, const json11::Json &solution) {
      auto obj = result.object_items();
      auto features = solution["features"].array_items();
      obj["features"] = features;
      return obj;
    }
  };

  /// Writes geo json to stream.
  void write(std::ostream &out,
             const vrp::models::Tasks &tasks,
             const LocationResolver &resolver) {
    out << createJson(tasks, resolver).dump();
  }

 private:
  /// Creates geo json for given tasks using location resolver provided.
  json11::Json createJson(const vrp::models::Tasks &tasks,
                          const LocationResolver &resolver) {
    int solutions = static_cast<int>(tasks.ids.size() / tasks.customers);
    json11::Json json = json11::Json::object {
        {"type", "FeatureCollection"},
        {"features", json11::Json::array()}
    };
    return thrust::transform_reduce(
        thrust::host,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(solutions),
        materialize_solution { tasks, resolver },
        json,
        merge_solution());
  }
};

}
}

#endif //VRP_STREAMS_GEOJSONWRITER_HPP
