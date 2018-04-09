#include "algorithms/Distances.cu"
#include "streams/input/SolomonReader.cu"
#include "streams/output/GeoJsonWriter.cu"

#include "solver/genetic/Populations.hpp"

#include <fstream>

using namespace vrp::algorithms;
using namespace vrp::genetic;
using namespace vrp::streams;

namespace {

struct LocationResolver final {
  explicit LocationResolver(std::fstream &in) {
    in.clear();
    in.seekg(0, std::ios::beg);

    // TODO read coordinates
  }

  std::pair<double, double> operator()(int customer) const {
    return coordinates.at(static_cast<unsigned long>(customer));
  }

 private:
  std::vector<std::pair<double, double>> coordinates;
};
}

int main(int argc, char* argv[]) {
  if (argc != 3)
    throw std::invalid_argument("Missing input or output argument.");

  std::fstream in(argv[1]);
  std::fstream out(argv[2]);

  auto problem = SolomonReader<geographic_distance>().read(in);
  auto solution = create_population(problem)({ 1 });

  GeoJsonWriter().write(out, solution, LocationResolver(in));

  return 0;
}
