#include "utils/validation/SolutionChecker.hpp"

using namespace vrp::models;
using namespace vrp::utils;

namespace {

/// Checks single individuum.
inline std::vector<std::string> check(const Solution& solution, int index) { return {}; }


}  // namespace

namespace vrp {
namespace utils {


SolutionChecker::Result SolutionChecker::check(const Solution& solution) const {
  return SolutionChecker::Result();
}


}  // namespace utils
}  // namespace vrp