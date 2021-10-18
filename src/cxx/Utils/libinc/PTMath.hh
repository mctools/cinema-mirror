#ifndef Prompt_Math_hh
#define Prompt_Math_hh

#include <vector>
#include <cmath>

namespace Prompt {
  bool floateq(double a, double b, double rtol=1e-14, double atol=1e-14);
  std::vector<double> logspace(double start, double stop, unsigned num);
  std::vector<double> linspace(double start, double stop, unsigned num);
}

#include "PTMath.icc"
#endif
