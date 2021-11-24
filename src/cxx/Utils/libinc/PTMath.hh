#ifndef Prompt_Math_hh
#define Prompt_Math_hh

#include "PromptCore.hh"

namespace Prompt {
  constexpr bool floateq(double a, double b, double rtol=1e-14, double atol=1e-14);
  std::vector<double> logspace(double start, double stop, unsigned num);
  std::vector<double> linspace(double start, double stop, unsigned num);

  constexpr double neutronEKin2k(double ekin);
  constexpr double neutronAngleCosine2Q(double anglecosine, double enin_eV, double enout_eV);
  constexpr double wl2ekin( double wl);
  constexpr double ekin2wl( double ekin);
  constexpr double wlsq2ekin( double wlsq );
  constexpr double ekin2wlsq( double ekin);
  constexpr double ekin2wlsqinv( double ekin);
}

#include "PTMath.icc"
#endif
