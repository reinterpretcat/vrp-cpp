#ifndef VRP_RUNTIME_CONFIG_HPP
#define VRP_RUNTIME_CONFIG_HPP

#include <thrust/execution_policy.h>

#define ANY_EXEC_UNIT __host__ __device__

#ifdef RUN_ON_DEVICE
#include "runtime/detail/device/Config.inl"
#else
#include "runtime/detail/host/Config.inl"
#endif

/// Define execution policy once.
const vrp::runtime::exec_unit_policy exec_unit = {};

#endif  // VRP_RUNTIME_CONFIG_HPP
