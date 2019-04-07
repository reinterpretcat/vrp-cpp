#if _MSC_VER
#define EXPORT_API __declspec(dllexport)
#elif _GCC
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT_API
#endif

#include "Solver.hpp"

#include <sstream>
#include <string>

/// Called when solution is created.
typedef void OnSuccess(const char* solution);

/// Called when solution cannot be created.
typedef void OnError(const char* error);

/// Executes function and catches exception if it occurs.
inline void
safeExecute(const std::function<void()>& action, OnError* onError) {
  try {
    action();
  } catch (std::exception& ex) { onError(ex.what()); }
}

extern "C" {
void EXPORT_API
solve(const char* problem, const char* format, OnSuccess* onSuccess, OnError* onError) {
  if (std::strcmp(format, "here") == 0) {
    // TODO
    onSuccess("Some result: TODO");
  } else {
    // TODO add all supported inputs
    onError("Not supported");
  }
}
}
