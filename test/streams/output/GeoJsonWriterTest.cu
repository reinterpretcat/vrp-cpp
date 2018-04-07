#include <catch/catch.hpp>

#include "streams/output/GeoJsonWriter.cu"

#include <thrust/fill.h>
#include <thrust/sequence.h>

using namespace vrp::models;
using namespace vrp::streams;

Tasks createSolution() {
  auto tasks = Tasks(6);

  thrust::sequence(tasks.ids.begin(), tasks.ids.end());
  thrust::fill(tasks.ids.begin(), tasks.ids.begin() + 3, 0);
  thrust::fill(tasks.ids.begin() + 3, tasks.ids.end(), 1);

  return tasks;
}

struct LocationResolver {

};

SCENARIO("Can write solution as geojson.", "[streams][geojson]") {
  LocationResolver resolver;
  GeoJsonWriter<LocationResolver> writer;

  writer.write(std::cout, createSolution(), resolver);

  // TODO
}