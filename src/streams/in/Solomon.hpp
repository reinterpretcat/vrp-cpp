#pragma once

#include "models/Problem.hpp"

#include <functional>
#include <istream>
#include <range/v3/all.hpp>
#include <sstream>

namespace vrp::streams::in {

/// Reads problem represented by classical solomon definition from stream.
template<typename DistanceCalculator>
struct read_solomon_type final {
  models::Problem operator()(std::istream& input) const { skipLines(input, 4); }

private:
  /// Skips selected amount of lines from stream.
  void skipLines(std::istream& input, int count) const {
    for (int i = 0; i < count; ++i)
      input.ignore(std::numeric_limits<std::streamsize>::max(), input.widen('\n'));
  }
};
}
