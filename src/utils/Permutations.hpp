#pragma once

#include <algorithm>
#include <range/v3/all.hpp>
#include <vector>

namespace vrp::utils {

/// Represents a range of integer permutations.
class permutation_range : public ranges::view_facade<permutation_range> {
  friend ranges::range_access;

  std::vector<int> data_;
  bool hasNext_;

  const std::vector<int>& read() const { return data_; }

  bool equal(ranges::default_sentinel_t) const { return !hasNext_; }

  void next() { hasNext_ = std::next_permutation(data_.begin(), data_.end()); }

public:
  permutation_range() = default;

  explicit permutation_range(int from, int to) : data_(), hasNext_(true) {
    data_ = ranges::view::iota(from, to + 1) | ranges::to_vector;
  }
};

/// Generates permutation of two sets in one index range.
/// TODO not memory efficient, needs improvements.
struct generate_set_permutations final {
  /// offset: offset of second range
  /// size: range size
  /// limit: result range max size
  std::vector<std::vector<int>> operator()(int offset, int size, int limit, std::mt19937& engine) {
    using namespace ranges;

    // NOTE memory allocations due to range requirements.
    auto seqFirst = permutation_range{0, offset} | to_vector;
    auto seqSecond = permutation_range{offset + 1, size - 1} | to_vector;
    auto prod = view::cartesian_product(seqFirst, seqSecond) | to_vector;

    return view::for_each(prod | view::sample(limit, engine),
                          [](const auto& tuple) {
                            return yield(view::concat(std::get<0>(tuple), std::get<1>(tuple)) | to_vector);
                          }) |
      to_vector;
  }
};
}