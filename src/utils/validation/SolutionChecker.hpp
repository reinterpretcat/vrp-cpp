#ifndef VRP_UTILS_SOLUTIONCHECKER_HPP
#define VRP_UTILS_SOLUTIONCHECKER_HPP

#include "models/Solution.hpp"

#include <string>
#include <vector>

namespace vrp {
namespace utils {

/// Checks whether solution is valid.
class SolutionChecker final {
 public:
  /// Represents checker result.
  struct Result final {
    /// Contains string representation of errors.
    std::vector<std::string> errors;
    /// Returns true if solution is valid.
    bool isValid() const { return errors.empty(); }
  };

  /// Preforms solution check.
  Result check(const vrp::models::Solution& solution) const;
};

}
}

#endif //VRP_SOLUTIONCHECKER_HPP
