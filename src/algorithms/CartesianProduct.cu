#ifndef VRP_ALGORITHMS_CARTESIANPRODUCTITERATOR_CU
#define VRP_ALGORITHMS_CARTESIANPRODUCTITERATOR_CU

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include <thrust/fill.h>
#include <thrust/device_vector.h>

namespace vrp {
namespace algorithms {

template<typename Iterator>
class repeated_range {
 public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct repeat_functor : public thrust::unary_function<difference_type, difference_type> {
    int repeats;

    repeat_functor(difference_type repeats) : repeats(repeats) {}

    __host__ __device__
    difference_type operator()(const difference_type &i) const {
      return i / repeats;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<repeat_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

  // type of the repeated_range iterator
  typedef PermutationIterator iterator;

  // construct repeated_range for the range [first,last)
  repeated_range(Iterator first, Iterator last, int repeats)
      : first(first), last(last), repeats(repeats) {}

  iterator begin(void) const {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
  }

  iterator end(void) const {
    return begin() + repeats * (last - first);
  }

 protected:
  int repeats;
  Iterator first;
  Iterator last;

};

template<typename Iterator>
class tiled_range {
 public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct tile_functor : public thrust::unary_function<difference_type, difference_type> {
    difference_type tile_size;

    tile_functor(difference_type tile_size) : tile_size(tile_size) {}

    __host__ __device__
    difference_type operator()(const difference_type &i) const {
      return i % tile_size;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<tile_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

  // type of the tiled_range iterator
  typedef PermutationIterator iterator;

  // construct repeated_range for the range [first,last)
  tiled_range(Iterator first, Iterator last, size_t tiles)
      : first(first), last(last), tiles(tiles) {}

  iterator begin(void) const {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
  }

  iterator end(void) const {
    return begin() + tiles * (last - first);
  }

 protected:
  size_t tiles;
  Iterator first;
  Iterator last;

};

}
}

#endif //VRP_ALGORITHMS_CARTESIANPRODUCTITERATOR_CU
