#ifndef Prompt_Math_hh
#define Prompt_Math_hh

#include "PromptCore.hh"

namespace Prompt {
  bool floateq(double a, double b, double rtol=1e-14, double atol=1e-14);
  std::vector<double> logspace(double start, double stop, unsigned num);
  std::vector<double> linspace(double start, double stop, unsigned num);

  double neutronEKin2k(double ekin);
  double neutronAngle2Q(double angle_rad, double enin_eV, double enout_eV);
}

#include "PTMath.icc"
#endif
