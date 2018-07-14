#ifndef VRP_RUNTIME_CONFIG_HPP
#define VRP_RUNTIME_CONFIG_HPP

/// Configuration to run on device.
#ifdef RUN_ON_DEVICE
#include "runtime/detail/device/Config.inl"
#else
#include "runtime/detail/host/Config.inl"
#endif


#endif  // VRP_ITERATORS_AGGREGATES_HPP
