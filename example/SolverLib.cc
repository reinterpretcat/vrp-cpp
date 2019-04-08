#if _MSC_VER
#define EXPORT_API __declspec(dllexport)
#elif _GCC
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT_API
#endif

#include "VarTypeSolver.hpp"

#include <sstream>

/// Called when solution is created.
typedef void OnSuccess(const char* solution);

/// Called when solution cannot be created.
typedef void OnError(const char* error);

extern "C" {
void EXPORT_API
solve(const char* problem, const char* inType, const char* outType, OnSuccess* onSuccess, OnError* onError) {
  try {
    std::stringstream inStream;
    inStream << problem;

    std::stringstream outStream;
    vrp::example::solve_based_on_type{}(inType, inStream, outType, outStream);

    auto solution = outStream.str();
    onSuccess(solution.data());
  } catch (std::exception& ex) {
    auto msg = std::string("Cannot solve: ") + ex.what();
    onError(msg.data());
  }
}
}
