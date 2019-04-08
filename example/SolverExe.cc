#include "VarTypeSolver.hpp"

#include <fstream>
#include <ostream>

using namespace vrp::algorithms;
using namespace vrp::algorithms::construction;
using namespace vrp::streams::in;

int
main(int argc, char* argv[]) {
  if (argc < 4) throw std::invalid_argument("Usage:\n\t$solver Path inType outType");

  auto inStream = std::fstream(argv[1], std::ios::in);
  vrp::example::solve_based_on_type{}(argv[2], inStream, argv[3], std::cout);

  return 0;
}
