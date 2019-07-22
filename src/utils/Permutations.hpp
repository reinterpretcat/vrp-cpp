#pragma once

#include <algorithm>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::utils {

/// Generates permutations lazily for a range of ints of given size.
/// NOTE allocates extra memory.
struct generate_permutations final {
  using Permutation = std::vector<int>;
  using Permutations = ranges::any_view<Permutation>;

  explicit generate_permutations(int size) : data_(), hasNext_(true) {
    data_ = ranges::view::iota(0, size) | ranges::to_vector;
  }

  Permutations operator()() {
    return ranges::view::for_each(  //
      ranges::view::ints | ranges::view::take_while([&](const auto) { return hasNext_; }),
      [&](const auto i) {
        if (i != 0) { hasNext_ = std::next_permutation(data_.begin(), data_.end()); }
        return ranges::yield(ranges::view::all(data_));
      });
  }

private:
  std::vector<int> data_;
  bool hasNext_;
};

}