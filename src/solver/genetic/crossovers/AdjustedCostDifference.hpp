#ifndef VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP
#define VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP

#include "models/Problem.hpp"
#include "models/Tasks.hpp"
#include "solver/genetic/Settings.hpp"

#include <thrust/pair.h>

namespace vrp {
namespace genetic {

/// Implements Adjusted Cost Difference Convolution crossover.
struct adjusted_cost_difference final {

  /// Represents a group of customers served together.
  struct Convolution final {
    int demand;
    int service;
    thrust::pair<int,int> customers;
    thrust::pair<int,int> times;
    thrust::pair<int,int> tasks;
  };

  /// Defines a context of operation.
  struct Context final {
    /// Holds individuum indicies to be processed.
    struct Generation {
      thrust::pair<int,int> parents;
      thrust::pair<int,int> offspring;
    };

    /// Holds solution data.
    struct Solution final {
      const vrp::models::Problem &problem;
      vrp::models::Tasks &tasks;

      Solution(const vrp::models::Problem &problem,
               vrp::models::Tasks &tasks) :
          problem(problem), tasks(tasks) {}
    };

    /// Defines object pool.
    struct Pool final {
      thrust::device_vector<float> differences;
      thrust::device_vector<bool> plan;
      thrust::device_vector<thrust::tuple<bool, int>> output;
      thrust::device_vector<int> lengths;
      thrust::device_vector<Convolution> convolutions;

      explicit Pool(size_t size) :
        differences(size), plan(size), output(size),
        lengths(size), convolutions(size) { }

      Pool(const Pool&) = delete;
      Pool &operator=(const Pool&) = delete;
    };

    Generation generation;
    Solution solution;
    Pool pool;

    Context(const Context::Generation &generation,
            const vrp::models::Problem &problem,
            vrp::models::Tasks &tasks) :
        generation(generation),
        solution(problem, tasks),
        pool(static_cast<size_t>(problem.size())) { }
  };

  void operator()(Context &context) const;
};

}
}

#endif //VRP_SOLVER_GENETIC_CROSSOVERS_ADJUSTED_COST_DIFFERENCE_HPP
