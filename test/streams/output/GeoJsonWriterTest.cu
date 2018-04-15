#include <catch/catch.hpp>

#include <json/json11.hpp>

#include "streams/output/GeoJsonWriter.hpp"

#include <sstream>
#include <thrust/fill.h>
#include <thrust/sequence.h>

using namespace vrp::models;
using namespace vrp::streams;

Tasks createSolution() {
  auto tasks = Tasks(6);

  thrust::sequence(tasks.ids.begin(), tasks.ids.end());
  thrust::fill(tasks.vehicles.begin(), tasks.vehicles.begin() + 3, 0);
  thrust::fill(tasks.vehicles.begin() + 3, tasks.vehicles.end(), 1);

  return tasks;
}

SCENARIO("Can write solution as geojson.", "[streams][geojson]") {
  GeoJsonWriter writer;
  std::stringstream ss;

  writer.write(ss, createSolution(), [](int customer) {
    return std::make_pair(customer, 0);
  });

  std::string err;
  auto json = json11::Json::parse(ss.str(), err, json11::JsonParse::STANDARD);
  REQUIRE(json["features"][0]["geometry"]["coordinates"].array_items().size() == 5);
  REQUIRE(json["features"][4]["geometry"]["coordinates"].array_items().size() == 4);
}