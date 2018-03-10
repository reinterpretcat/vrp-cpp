#include "solver/genetic/init/PopulationFactory.hpp"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

using namespace vrp::models;
using namespace vrp::genetic;


namespace {

/// Prepare solution by setting all tasks as unprocessed within customer.
struct InitPopulation {
  int problemSize;

  __host__ __device__
  void operator()(thrust::tuple<int, int&, int&> tuple) {
    int i = thrust::get<0>(tuple) % problemSize;
    thrust::get<1>(tuple) = i;
    thrust::get<2>(tuple) = -1;
  }
};

}

namespace vrp {
namespace genetic {

Tasks createPopulation(const Problem &problem,
                       const Resources &resources,
                       const Settings &settings) {

  Tasks population { settings.populationSize *  problem.size() };

  thrust::for_each(thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(0),
          population.ids.begin(),
          population.vehicles.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          thrust::make_counting_iterator(population.size()),
          population.ids.end(),
          population.vehicles.end())),
      InitPopulation { problem.size() }
  );

  return std::move(population);
}

}
}