#ifndef VRP_RUNTIME_CONFIG_HPP
#define VRP_RUNTIME_CONFIG_HPP

#include <thrust/execution_policy.h>


#ifdef USE_CUDA_BACKEND
#include "runtime/detail/cuda/Config.inl"

#elif USE_OMP_BACKEND
#include "runtime/detail/omp/Config.inl"

#elif USE_CPP_BACKEND
#include "runtime/detail/cpp/Config.inl"
#endif

/// Define execution policy once.
static const vrp::runtime::exec_unit_policy exec_unit{};

#endif  // VRP_RUNTIME_CONFIG_HPP
