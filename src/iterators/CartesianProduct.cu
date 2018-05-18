#ifndef VRP_ITERATORS_CARTESIANPRODUCTITERATOR_CU
#define VRP_ITERATORS_CARTESIANPRODUCTITERATOR_CU

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include <thrust/fill.h>
#include <thrust/device_vector.h>

namespace vrp {
namespace iterators {

template<typename Iterator>
class repeated_range final {
 public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct repeat_functor : public thrust::unary_function<difference_type, difference_type> {
    int repeats;

    __host__ __device__
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
  __host__ __device__
  repeated_range(Iterator first, Iterator last, size_t repeats)
      : first(first), last(last), repeats(repeats) {}

  __host__ __device__
  iterator begin(void) const {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
  }

  __host__ __device__
  iterator end(void) const {
    return begin() + repeats * (last - first);
  }

 protected:
  size_t repeats;
  Iterator first;
  Iterator last;

};

template<typename Iterator>
class tiled_range final {
 public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct tile_functor : public thrust::unary_function<difference_type, difference_type> {
    difference_type tile_size;

    __host__ __device__
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
  __host__ __device__
  tiled_range(Iterator first, Iterator last, size_t tiles)
      : first(first), last(last), tiles(tiles) {}

  __host__ __device__
  iterator begin(void) const {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
  }

  __host__ __device__
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

#endif //VRP_ITERATORS_CARTESIANPRODUCTITERATOR_CU
