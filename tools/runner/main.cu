#include "Resolvers.hpp"
#include "algorithms/distances/Cartesian.hpp"
#include "algorithms/distances/Geographic.hpp"
#include "algorithms/genetic/Evolution.hpp"
#include "algorithms/genetic/Strategies.hpp"
#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "streams/input/SolomonReader.hpp"
#include "streams/output/GeoJsonWriter.hpp"
#include "streams/output/MatrixTextWriter.hpp"

#include <ostream>
#include <thrust/host_vector.h>

using namespace vrp::algorithms::distances;
using namespace vrp::algorithms::genetic;
using namespace vrp::models;
using namespace vrp::streams;
using namespace vrp::utils;

namespace {
const int PopulationSize = 12;

template<typename Distance, typename Mapper>
void solve(std::fstream& in, std::fstream& out, const Distance& distance, const Mapper& mapper) {
  auto problem = SolomonReader().read(in, distance);
  auto ctx = run_evolution<GuidedStrategy>{{PopulationSize}}(std::move(problem));

  MatrixTextWriter().write(std::cout, ctx.solution);

  GeoJsonWriter().write(out, ctx.solution.tasks, location_resolver<decltype(mapper)>(in, mapper),
                        static_cast<thrust::pair<int, float>>(ctx.costs.front()).first);
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) throw std::invalid_argument("Missing input or output argument.");

  std::fstream in(argv[1], std::fstream::in);
  std::fstream out(argv[2], std::fstream::out);
  auto isGeo = argc > 3;
  if (isGeo) {
    solve(in, out, geographic_distance<>(),
          BoundingBoxMapper({{13.3285, 52.4915}, {13.4663, 52.5553}}));
  } else {
    solve(in, out, cartesian_distance(), DefaultMapper(1));
  }

  return 0;
}
